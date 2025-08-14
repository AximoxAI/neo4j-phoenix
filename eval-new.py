import dspy
import os
import pandas as pd
from phoenix.otel import register
from phoenix.trace import SpanEvaluations
from phoenix.trace.dsl import SpanQuery
from openinference.instrumentation.dspy import DSPyInstrumentor
from datasets import load_dataset
from neo4j import GraphDatabase
from opentelemetry import trace
from sentence_transformers import SentenceTransformer, util
import time
import logging

from openinference.instrumentation import suppress_tracing
from phoenix.client import Client

# Setup simple logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def init_phoenix():
    """Initialize Phoenix UI for observability."""
    PROJECT_NAME = "default"
    tracer_provider = register(
        project_name=PROJECT_NAME,
        auto_instrument=True,
        endpoint="http://0.0.0.0:6006/v1/traces"
    )

    tracer = tracer_provider.get_tracer(__name__)
    DSPyInstrumentor().instrument()
    logger.info("DSPy instrumentation enabled")

    client = Client()    

    return tracer, client


# Configuration
CONFIG = {
    "neo4j": {
        "uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        "user": os.getenv("NEO4J_USER", "neo4j"),
        "password": os.getenv("NEO4J_PASSWORD", "login123")
    },
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


def setup_neo4j():
    """Setup Neo4j connection with proper error handling."""
    logger.info("Setting up Neo4j connection")
    try:
        driver = GraphDatabase.driver(
            CONFIG["neo4j"]["uri"], 
            auth=(CONFIG["neo4j"]["user"], CONFIG["neo4j"]["password"])
        )
        driver.verify_connectivity()
        logger.info("Successfully connected to Neo4j")
        return driver
    except Exception as e:
        logger.warning(f"Failed to connect to Neo4j: {e}. Cypher execution will be skipped.")
        return None


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


def compare_query_results(ground_truth_query, generated_query, driver):
    """Executes both queries and compares their results (no tracing during evaluation)."""
    logger.debug("Comparing query results")
    # Suppress tracing during evaluation
    with suppress_tracing():
        if not driver:
            logger.warning("Neo4j driver not available for query comparison")
            return {"correct": None, "reason": "Neo4j connection unavailable"}
            
        gt_results, gen_results, gt_error, gen_error = [], [], None, None
        
        # Execute ground truth query
        try:
            with driver.session() as session:
                gt_results = session.run(ground_truth_query).data()
            logger.debug(f"Ground truth query executed successfully, {len(gt_results)} results")
        except Exception as e:
            gt_error = str(e)
            logger.error(f"Ground truth query error: {gt_error}")

        # Execute generated query  
        try:
            with driver.session() as session:
                gen_results = session.run(generated_query).data()
            logger.debug(f"Generated query executed successfully, {len(gen_results)} results")
        except Exception as e:
            gen_error = str(e)
            logger.error(f"Generated query error: {gen_error}")
            
        # Compare results
        if gen_error:
            result = {"correct": False, "reason": f"Generated query error: {gen_error}"}
        elif gt_error:
            result = {"correct": None, "reason": f"Ground truth query error: {gt_error}"}
        else:
            # Normalize results for comparison
            canonical_gt = sorted([tuple(sorted(d.items())) for d in gt_results])
            canonical_gen = sorted([tuple(sorted(d.items())) for d in gen_results])
            
            if canonical_gt == canonical_gen:
                result = {"correct": True, "reason": "Results match"}
                logger.debug("Query results match")
            else:
                result = {"correct": False, "reason": "Results mismatch"}
                logger.debug("Query results mismatch")
        
        return result


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
                                sentence_model, target_language, driver, sample_id):
    """Main function to process one sample, orchestrating all steps."""
    logger.debug(f"Processing sample {sample_id} for language {target_language}")
    
    # Create a custom span for the evaluation process
    with tracer.start_as_current_span(
        "multilingual_evaluation", 
        attributes={
            "evaluation.question": sample['question'][:200],  # Truncate to avoid issues
            "evaluation.target_language": target_language,
            "evaluation.sample_id": str(sample_id),
            "evaluation.sample_type": "multilingual_cypher_generation"
        }
    ) as eval_span:
        
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
            
            # Evaluate Cypher query correctness
            cypher_assessment = compare_query_results(
                sample['cypher'], pipeline_result.generated_query, driver
            )
            
            # Set evaluation results as attributes
            evaluation_attrs = {
                "evaluation.translation_score": float(translation_score_result.get('score', 0.0)),
                "evaluation.cypher_correct": cypher_assessment.get('correct'),
                "evaluation.cypher_reason": cypher_assessment.get('reason', ''),
                "evaluation.back_translated_text": translation_score_result.get('back_translated_text', ''),
                "evaluation.overall_success": (
                    cypher_assessment.get('correct') and
                    translation_score_result.get('score', 0.0) > 0.7
                ),
                "evaluation.ground_truth_query": sample['cypher'][:200],  # Truncate
                "evaluation.generated_query": pipeline_result.generated_query[:200],  # Truncate
                "evaluation.original_text": sample['question'][:200],  # Truncate
                "evaluation.translated_text": pipeline_result.translated_question[:200]  # Truncate
            }
            
            # Safely set attributes
            safe_set_span_attributes(eval_span, evaluation_attrs)
            
            logger.debug(f"Sample {sample_id} evaluation completed successfully")
            return pipeline_result, translation_score_result, cypher_assessment
            
        except Exception as e:
            logger.error(f"Error processing sample {sample_id}: {e}")
            # Safely set error attributes
            safe_set_span_attributes(eval_span, {
                "error": str(e),
                "evaluation.pipeline_error": str(e)
            })
            return None, {"score": 0.0, "error": str(e)}, {"correct": False, "reason": f"Pipeline error: {e}"}


def query_and_upload_evaluations(model_name, client):
    """Query spans and upload evaluations to Phoenix."""
    logger.info(f"Querying spans for model {model_name}")
    
    # Wait for spans to be processed
    time.sleep(3)
    
    try:
        # Query spans for evaluation
        eval_query = SpanQuery().where(
            "attributes.`evaluation.sample_type` == 'multilingual_cypher_generation'"
        ).select(
            span_id="span_id",
            question="attributes.`evaluation.question`",
            target_language="attributes.`evaluation.target_language`",
            sample_id="attributes.`evaluation.sample_id`",
            translation_score="attributes.`evaluation.translation_score`",
            original_text="attributes.`evaluation.original_text`",
            translated_text="attributes.`evaluation.translated_text`",
            back_translated_text="attributes.`evaluation.back_translated_text`",
            cypher_correct="attributes.`evaluation.cypher_correct`",
            cypher_reason="attributes.`evaluation.cypher_reason`",
            ground_truth_query="attributes.`evaluation.ground_truth_query`",
            generated_query="attributes.`evaluation.generated_query`",
            overall_success="attributes.`evaluation.overall_success`"
        )
        
        eval_df = client.query_spans(eval_query, project_name=project_name, timeout=30)
        
        if eval_df.empty:
            logger.warning("No evaluation spans found")
            return
            
        logger.info(f"Found {len(eval_df)} evaluation spans")
        
        # Upload translation quality evaluations
        translation_df = eval_df.dropna(subset=["translation_score"]).copy()
        if not translation_df.empty:
            translation_df['score'] = translation_df['translation_score'].astype(float)
            translation_df['label'] = translation_df['score'].apply(
                lambda x: 'excellent' if x > 0.9 else 'good' if x > 0.7 else 'fair' if x > 0.5 else 'poor'
            )
            translation_df['explanation'] = translation_df.apply(
                lambda row: f"Translation quality score: {row['score']:.3f}",
                axis=1
            )
            
            client.log_evaluations(
                SpanEvaluations(
                    eval_name=f"translation_quality_{model_name}",
                    dataframe=translation_df[['span_id', 'score', 'label', 'explanation']]
                )
            )
            logger.info(f"Uploaded translation quality evaluations for {len(translation_df)} spans")
        
        # Upload Cypher correctness evaluations  
        cypher_df = eval_df.dropna(subset=["cypher_correct"]).copy()
        if not cypher_df.empty:
            cypher_df['score'] = cypher_df['cypher_correct'].apply(
                lambda x: 1.0 if x == True else 0.5 if pd.isna(x) else 0.0
            )
            cypher_df['label'] = cypher_df['cypher_correct'].apply(
                lambda x: 'correct' if x == True else 'unknown' if pd.isna(x) else 'incorrect'
            )
            cypher_df['explanation'] = cypher_df['cypher_reason'].fillna("Unknown evaluation result")
            
            client.log_evaluations(
                SpanEvaluations(
                    eval_name=f"cypher_correctness_{model_name}",
                    dataframe=cypher_df[['span_id', 'score', 'label', 'explanation']]
                )
            )
            logger.info(f"Uploaded Cypher correctness evaluations for {len(cypher_df)} spans")
            
        # Upload overall success evaluations
        overall_df = eval_df.dropna(subset=["overall_success"]).copy()
        if not overall_df.empty:
            overall_df['score'] = overall_df['overall_success'].apply(
                lambda x: 1.0 if x == True else 0.0
            )
            overall_df['label'] = overall_df['overall_success'].apply(
                lambda x: 'success' if x == True else 'failure'
            )
            overall_df['explanation'] = "Combined evaluation of translation quality (>0.7) and Cypher correctness"
            
            client.log_evaluations(
                SpanEvaluations(
                    eval_name=f"overall_success_{model_name}",
                    dataframe=overall_df[['span_id', 'score', 'label', 'explanation']]
                )
            )
            logger.info(f"Uploaded overall success evaluations for {len(overall_df)} spans")
            
    except Exception as e:
        logger.error(f"Error querying/uploading evaluations: {e}")


def main():
    """Main evaluation function."""
    logger.info("Starting multilingual Cypher evaluation")
    
    # Initialize Phoenix
    tracer, client = init_phoenix()
    
    # Validate setup
    validate_api_keys()
    
    # Setup resources
    neo4j_driver = setup_neo4j()
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
            
            # Test each target language
            for target_language in CONFIG["target_languages"]:
                logger.info(f"Testing language: {target_language}")
                
                for i, sample in enumerate(train_samples):
                    logger.info(f"Processing sample {i+1}/{len(train_samples)}")
                    
                    pipeline_result, translation_score, cypher_assessment = process_and_evaluate_sample(
                        tracer=tracer, sample=sample, evaluator=evaluator,
                        back_translator=back_translator, sentence_model=sentence_model,
                        target_language=target_language, driver=neo4j_driver, sample_id=i
                    )
                    
                    # Store results for summary reporting
                    result = {
                        "model_name": model_name,
                        "sample_id": i,
                        "question": sample['question'],
                        "target_language": target_language,
                        "translation_score": translation_score.get('score', 0.0),
                        "is_correct": cypher_assessment.get('correct'),
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
            query_and_upload_evaluations(model_name, client)
    
    finally:
        # Cleanup
        if neo4j_driver:
            logger.info("Closing Neo4j connection")
            neo4j_driver.close()
    
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
        cypher_accuracy_pct=('is_correct', lambda x: (x == True).mean() * 100),
        error_rate_pct=('has_error', lambda x: x.mean() * 100)
    ).round(2).reset_index()
    
    logger.info("Overall Model Performance:")
    print(summary.to_string(index=False))
    
    # Performance by language
    lang_summary = df.groupby(['target_language']).agg(
        avg_translation_score=('translation_score', 'mean'),
        cypher_accuracy_pct=('is_correct', lambda x: (x == True).mean() * 100)
    ).round(2).reset_index()
    
    logger.info("Performance by Language:")
    print(lang_summary.to_string(index=False))
    
    logger.info("All evaluations uploaded to Phoenix!")
    print(f"View detailed traces and evaluations in Phoenix UI: {session.url}")
    print("\nIn Phoenix, you can:")
    print("- Filter spans by evaluation.sample_type")
    print("- View translation quality scores and labels")
    print("- Analyze Cypher correctness by model and language")
    print("- Compare overall success rates across models")


if __name__ == "__main__":
    main()