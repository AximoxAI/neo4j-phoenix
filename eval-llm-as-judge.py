from dotenv import load_dotenv
import dspy
import os
import pandas as pd
from phoenix.otel import register
from openinference.instrumentation.dspy import DSPyInstrumentor
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
from openinference.semconv.trace import SpanAttributes
import logging


from openinference.instrumentation import suppress_tracing
from phoenix.client import Client
from opentelemetry.trace import Status, StatusCode
from opentelemetry.trace import format_span_id
from openinference.instrumentation.litellm import LiteLLMInstrumentor

# Setup simple logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


load_dotenv()


def init_phoenix():
    """Initialize Phoenix UI for observability."""
    PROJECT_NAME = "default"
    tracer_provider = register(
        project_name=PROJECT_NAME,
        auto_instrument=True,
        endpoint="http://0.0.0.0:6006/v1/traces"
    )

    tracer = tracer_provider.get_tracer(__name__)
    
    DSPyInstrumentor().instrument(tracer_provider=tracer_provider)
    LiteLLMInstrumentor().instrument(tracer_provider=tracer_provider)

    logger.info("DSPy instrumentation enabled")

    client = Client(base_url="http://localhost:6006")

    return tracer, client


# Configuration
CONFIG = {
    "dataset_size": int(os.getenv("DATASET_SIZE", "10")),
    "target_languages": ["Hindi", "Tamil", "Telugu"]
}

# Define all models to be evaluated
models_to_evaluate = [
    {
        "name": "openai/gpt-oss-20b",
        "instance": dspy.LM(model='openai/gpt-oss-20b', api_base="http://localhost:1234/v1"),
        "api_key_env": "OPENAI_API_KEY"

    }
]


def validate_api_keys():
    """Validate that required API keys are present."""
    logger.info("Validating API keys")
    missing_keys = []
    for model in models_to_evaluate:
        api_key = model["api_key_env"]
        if not os.getenv(api_key):
            missing_keys.append(api_key)
            logger.warning(f"Missing API key: {api_key}")
    
    if missing_keys:
        logger.error(f"Missing API keys: {missing_keys}")
        logger.error("Some models may fail to initialize")
    else:
        logger.info("All API keys validated successfully")


def load_resources():
    """Load dataset and sentence model."""
    logger.info("Loading text2cypher dataset")
    ds = load_dataset("neo4j/text2cypher-2024v1")
    train_samples = ds['train'].select(range(CONFIG["dataset_size"]))
    logger.info(f"Dataset loaded with {len(train_samples)} samples")

    logger.info("Loading sentence-transformer model for semantic scoring")
    
    # using all-MiniLM-L6-v2
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("Sentence model loaded successfully")
    
    return train_samples, sentence_model


class Translate(dspy.Signature):
    """Translates text into a target language."""
    source_text = dspy.InputField(desc="The question in source language")
    target_language = dspy.InputField(desc="Name of the language")
    translated_text = dspy.OutputField(desc="The question in the target language.")


class GenerateCypher(dspy.Signature):
    """Generates a Cypher query from a question and schema."""
    question = dspy.InputField(desc="Natural Language Question")
    neo4j_schema = dspy.InputField(desc="Neo4J schema")
    cypher_query = dspy.OutputField(desc="Generated Cypher Query")


class EvaluateCypherEquivalence(dspy.Signature):
    """Determines if two Cypher queries are functionally equivalent using categorical assessment.
       If generated query is able to answer the english query it is Equivalent. Queries maybe returning additional 
       fields, but if query is able to answer the english query then it should be considered equivalent.
    """
    ground_truth_query = dspy.InputField(desc="The correct/reference Cypher query")
    generated_query = dspy.InputField(desc="The generated Cypher query to evaluate")
    natural_lang_query = dspy.InputField(desc="Natural Language Query in english")

    schema = dspy.InputField(desc="Neo4j database schema for context")
    equivalence_category = dspy.OutputField(desc="One of: EQUIVALENT, PARTIALLY_CORRECT, INCORRECT, SYNTAX_ERROR")
    reasoning = dspy.OutputField(desc="Brief explanation of the categorization")


class FullPipelineEvaluator(dspy.Module):
    """A DSPy module that runs the translation and Cypher generation steps."""
    def __init__(self):
        super().__init__()
        self.translator = dspy.Predict(Translate)
        self.cypher_generator = dspy.Predict(GenerateCypher)

    def forward(self, question, schema, target_language):
        logger.debug(f"Running pipeline for question: {question[:50]}...")
        
        # Get translation
        t = self.translator(source_text=question, target_language=target_language)
        logger.debug(f"Translation completed: {t.translated_text[:50]}...")
        
        # Generate cypher
        c = self.cypher_generator(question=t.translated_text, neo4j_schema=schema)
        logger.debug(f"Cypher generation completed: {c.cypher_query[:50]}...")
        
        return dspy.Prediction(
            translated_question=t.translated_text, 
            generated_query=c.cypher_query
        )


# --- Evaluation Functions ---
def score_translation_quality(original_text, translated_text, back_translator, sentence_model):
    """Performs back-translation and returns a semantic similarity score."""
    logger.debug("Evaluating translation quality")
    # Suppress tracing during evaluation to avoid creating evaluation spans
    with suppress_tracing():
        try:
            back_translation_result = back_translator(
                source_text=translated_text,
                target_language="English"
            )
            back_translated_text = back_translation_result.translated_text
            
            embedding_1 = sentence_model.encode(original_text, convert_to_tensor=True)
            embedding_2 = sentence_model.encode(back_translated_text, convert_to_tensor=True)
            
            similarity_score = util.pytorch_cos_sim(embedding_1, embedding_2).item()
            logger.debug(f"Translation quality score: {similarity_score:.3f}")
            
            return {"score": similarity_score, "back_translated_text": back_translated_text}
        except Exception as e:
            logger.error(f"Error in translation quality evaluation: {e}")
            return {"score": 0.0, "back_translated_text": "", "error": str(e)}


def compare_query_results_with_llm(ground_truth_query, generated_query, schema, cypher_judge,
                                   natural_language_question):
    """Compare queries using categorical LLM judge."""
    logger.debug("Comparing queries using categorical LLM judge")
    
    # with suppress_tracing():
    try:
        result = cypher_judge(
                ground_truth_query=ground_truth_query,
                generated_query=generated_query,
                schema=schema,
                natural_lang_query=natural_language_question
            )
            
        # Map categories to boolean and score
        category = result.equivalence_category.upper()
            
        if category == "EQUIVALENT":
            correct = True
            score = 1.0
        elif category == "PARTIALLY_CORRECT":
            correct = None  # Partial credit
            score = 0.5
        elif category in ["INCORRECT", "SYNTAX_ERROR"]:
            correct = False
            score = 0.0
        else:
            # Fallback for unexpected categories
            correct = False
            score = 0.0
                
        return {
                "correct": correct,
                "category": category,
                "score": score,
                "reason": result.reasoning
            }
            
    except Exception as e:
        logger.error(f"Error in LLM judge evaluation: {e}")
        return {
            "correct": False,
            "category": "ERROR",
            "score": 0.0,
            "reason": f"Judge evaluation error: {e}"
        }


def safe_set_span_attributes(span, attributes):
    """Safely set span attributes without causing warnings."""
    if span and not span.is_recording():
        logger.debug("Span is not recording, skipping attribute setting")
        return
    try:
        if span and hasattr(span, 'set_attributes'):
            span.set_attributes(attributes)
        elif span:
            for key, value in attributes.items():
                span.set_attribute(key, value)
    except Exception as e:
        logger.debug(f"Could not set span attributes: {e}")


def process_and_evaluate_sample(tracer, sample, evaluator, back_translator,
                                sentence_model, target_language, cypher_judge, sample_id):
    """Main function to process one sample, orchestrating all steps."""
    logger.debug(f"Processing sample {sample_id} for language {target_language}")
    
    # Create a custom span for the evaluation process
    with tracer.start_as_current_span(
        "multilingual_evaluation_" + target_language,
        openinference_span_kind="chain",
        attributes={
            "evaluation.question": sample['question'][:200],
            "evaluation.target_language": target_language,
            "evaluation.sample_id": str(sample_id),
            "evaluation.sample_type": "multilingual_cypher_generation"
        }
    ) as eval_span:
        eval_span.set_status(Status(StatusCode.OK))
        try:
            # Store data for later attribute setting
            results_data = {
                "question": sample['question'],
                "schema": sample['schema'],
                "ground_truth_cypher": sample['cypher'],
                "target_language": target_language,
                "sample_id": sample_id
            }
            
            # Run the full pipeline
            logger.info("Question in English")
            logger.info(sample['question'])
            
            pipeline_result = evaluator(
                question=sample['question'],
                schema=sample['schema'],
                target_language=target_language
            )
            logger.debug("Pipeline execution completed")
            
            # Store pipeline results
            results_data.update({
                "translated_question": pipeline_result.translated_question,
                "generated_query": pipeline_result.generated_query
            })

            # Evaluate translation quality
            translation_score_result = score_translation_quality(
                sample['question'], pipeline_result.translated_question, 
                back_translator, sentence_model
            )
            
            logger.info("Translated text")
            logger.info(pipeline_result.translated_question)
            
            # Evaluate Cypher query correctness using LLM judge
            cypher_assessment = compare_query_results_with_llm(
                sample['cypher'], pipeline_result.generated_query, sample['schema'], cypher_judge, sample['question']
            )
            
            # Set evaluation results as attributes
            evaluation_attrs = {
                "evaluation.translation_score": float(translation_score_result.get('score', 0.0)),
                "evaluation.cypher_correct": cypher_assessment.get('correct'),
                "evaluation.cypher_category": cypher_assessment.get('category', ''),
                "evaluation.cypher_score": cypher_assessment.get('score', 0.0),
                "evaluation.cypher_reason": cypher_assessment.get('reason', ''),
                "evaluation.back_translated_text": translation_score_result.get('back_translated_text', ''),
                "evaluation.overall_success": (
                    cypher_assessment.get('correct') and
                    translation_score_result.get('score', 0.0) > 0.7
                ),
                "evaluation.ground_truth_query": sample['cypher'][:200],  # Truncate
                "evaluation.generated_query": pipeline_result.generated_query[:200],  # Truncate
                "evaluation.original_text": sample['question'][:200],  # Truncate
                "evaluation.translated_text": pipeline_result.translated_question[:200],  # Truncate
                SpanAttributes.INPUT_VALUE: pipeline_result.translated_question[:200],
                SpanAttributes.OUTPUT_VALUE: pipeline_result.generated_query[:200],
            }
            
            # Safely set attributes
            safe_set_span_attributes(eval_span, evaluation_attrs)
            eval_span.set_attribute("feedback", cypher_assessment.get('category', ''))

            span_id = format_span_id(eval_span.get_span_context().span_id)

            logger.debug(f"Sample {sample_id} evaluation completed successfully")
            return pipeline_result, translation_score_result, cypher_assessment, span_id

        except Exception as e:
            logger.error(f"Error processing sample {sample_id}: {e}")
            # Safely set error attributes
            safe_set_span_attributes(eval_span, {
                "error": str(e),
                "evaluation.pipeline_error": str(e)
            })
            eval_span.set_status(Status(StatusCode.ERROR))
            return None, {"score": 0.0, "error": str(e)}, {"correct": False, "reason": f"Pipeline error: {e}"}, span_id
        

def main():
    """Main evaluation function."""
    logger.info("Starting multilingual Cypher evaluation")
    
    # Initialize Phoenix
    tracer, client = init_phoenix()   
    # Validate setup
    validate_api_keys()
    
    # Setup resources
    train_samples, sentence_model = load_resources()
    
    all_results = []
    
    try:
        for model_config in models_to_evaluate:
            model_name = model_config['name']
            llm_instance = model_config['instance']
            
            logger.info(f"Evaluating Model: {model_name}")        
            # Configure DSPy with current model
            dspy.settings.configure(lm=llm_instance)
            
            # Initialize components
            evaluator = FullPipelineEvaluator()
            back_translator = dspy.Predict(Translate)
            cypher_judge = dspy.Predict(EvaluateCypherEquivalence)
            
            # Test each target language
            for target_language in CONFIG["target_languages"]:
                logger.info(f"Testing language: {target_language}")
                
                for i, sample in enumerate(train_samples):
                    logger.info(f"Processing sample {i+1}/{len(train_samples)}")
                    
                    pipeline_result, translation_score, cypher_assessment, span_id = process_and_evaluate_sample(
                        tracer=tracer, sample=sample, evaluator=evaluator,
                        back_translator=back_translator, sentence_model=sentence_model,
                        target_language=target_language, cypher_judge=cypher_judge, sample_id=i
                    )

                    # logger.info("Saving span data for span id:", span_id)
                    try:
                        client.annotations.add_span_annotation(
                            annotation_name="translation_score",
                            annotator_kind="LLM",
                            span_id=span_id,
                            label="translation",
                            score=float(translation_score.get('score', 0.0))
                        )
                    except Exception as e:
                        logger.exception("exception occurred while pushing span transalation {e}", e)

                    try:
                        client.annotations.add_span_annotation(
                            annotation_name="cypher_score",
                            annotator_kind="LLM",
                            span_id=span_id,
                            label="cypher",
                            score=cypher_assessment.get('score', 0.0),
                            metadata={"category": cypher_assessment.get('category', '')}
                        )
                    except Exception as e:
                        logger.exception("exception occurred while pushing span cypher assessment {e}", e) 
                    
                    # Store results for summary reporting
                    result = {
                        "model_name": model_name,
                        "sample_id": i,
                        "question": sample['question'],
                        "target_language": target_language,
                        "translation_score": translation_score.get('score', 0.0),
                        "is_correct": cypher_assessment.get('correct'),
                        "cypher_category": cypher_assessment.get('category'),
                        "cypher_score": cypher_assessment.get('score', 0.0),
                        "reason": cypher_assessment.get('reason'),
                        "has_error": 'error' in translation_score or 'error' in cypher_assessment
                    }
                    
                    if pipeline_result:
                        result.update({
                            "translated_question": pipeline_result.translated_question,
                            "generated_query": pipeline_result.generated_query
                        })
                    
                    all_results.append(result)
            
            # Query spans and upload evaluations for this model
            logger.info(f"Uploading evaluations to Phoenix for model {model_name}")
            # query_and_upload_evaluations(model_name, client, project_name)
    
    finally:
        # No cleanup needed since Neo4j is removed
        pass
    
    # Generate summary report
    logger.info("Generating final evaluation report")
    if not all_results:
        logger.error("No results to analyze. Check your API keys and configuration.")
        return
    
    df = pd.DataFrame(all_results)
    
    # Overall performance by model
    summary = df.groupby(['model_name']).agg(
        samples_processed=('sample_id', 'count'),
        avg_translation_score=('translation_score', 'mean'),
        avg_cypher_score=('cypher_score', 'mean'),
        cypher_accuracy_pct=('is_correct', lambda x: (x == True).mean() * 100),
        error_rate_pct=('has_error', lambda x: x.mean() * 100)
    ).round(2).reset_index()
    
    logger.info("Overall Model Performance:")
    print(summary.to_string(index=False))
    
    # Performance by language
    lang_summary = df.groupby(['target_language']).agg(
        avg_translation_score=('translation_score', 'mean'),
        avg_cypher_score=('cypher_score', 'mean'),
        cypher_accuracy_pct=('is_correct', lambda x: (x == True).mean() * 100)
    ).round(2).reset_index()
    
    logger.info("Performance by Language:")
    print(lang_summary.to_string(index=False))
    
    # Category distribution
    if 'cypher_category' in df.columns:
        category_counts = df['cypher_category'].value_counts()
        logger.info("Cypher Evaluation Categories:")
        print(category_counts.to_string())
    
    logger.info("All evaluations uploaded to Phoenix!")
    print("View detailed traces and evaluations in Phoenix UI")
    print("\nIn Phoenix, you can:")
    print("- Filter spans by evaluation.sample_type")
    print("- View translation quality scores and labels")
    print("- Analyze Cypher correctness by categorical evaluation")
    print("- Compare overall success rates across models")


if __name__ == "__main__":
    main()