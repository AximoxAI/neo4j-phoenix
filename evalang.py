import dspy
import os
import pandas as pd
import phoenix as px
from phoenix.otel import register
from openinference.instrumentation.dspy import DSPyInstrumentor
from datasets import load_dataset
from neo4j import GraphDatabase
from opentelemetry import trace
from sentence_transformers import SentenceTransformer, util


def init_phoenix():
    # Launch the Phoenix UI. The session URL will be printed.
    session = px.launch_app()
    print(f"Phoenix UI running at: {session.url}")
    PROJECT_NAME = "multilingual_benchmark"
    tracer_provider = register(
        project_name=PROJECT_NAME,
        auto_instrument=True,
    )

    tracer = tracer_provider.get_tracer(__name__)

    # Instrument DSPy to automatically trace all LLM calls.
    DSPyInstrumentor().instrument()


# Define all models to be evaluated
models_to_evaluate = [
    {
        "name": "gpt-4o",
        "instance": dspy.OpenAI(model='gpt-4o', max_tokens=1000),
        "api_key_env": "OPENAI_API_KEY"
    },
    {
        "name": "gpt-3.5-turbo",
        "instance": dspy.OpenAI(model='gpt-3.5-turbo', max_tokens=1000),
        "api_key_env": "OPENAI_API_KEY"
    },
    {
        "name": "claude-3-haiku",
        "instance": dspy.Anthropic(model='claude-3-haiku-20240307', max_tokens=1000),
        "api_key_env": "ANTHROPIC_API_KEY"
    },
    {
        "name": "gemini-1.5-flash",
        "instance": dspy.Google(model='models/gemini-1.5-flash-latest', max_tokens=1000),
        "api_key_env": "GOOGLE_API_KEY"
    }
]


NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "your_neo4j_password"

try:
    neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    neo4j_driver.verify_connectivity()
    print("Successfully connected to Neo4j.")
except Exception as e:
    print(f"Failed to connect to Neo4j: {e}. Cypher execution will be skipped.")
    neo4j_driver = None

print("Loading text2cypher dataset...")
ds = load_dataset("neo4j/text2cypher-2024v1")
# Using a small subset for a quick demonstration run
train_samples = ds['train'].select(range(5))
print("Dataset loaded.")

print("\nLoading sentence-transformer model for semantic scoring...")
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Sentence model loaded.")


# --- 3. Define DSPy Modules and Helper Functions ---

class Translate(dspy.Signature):
    """Translates text into a target language."""
    source_text = dspy.InputField()
    target_language = dspy.InputField()
    translated_text = dspy.OutputField()


class GenerateCypher(dspy.Signature):
    """Generates a Cypher query from a question and schema."""
    question = dspy.InputField()
    neo4j_schema = dspy.InputField()
    cypher_query = dspy.OutputField()


class FullPipelineEvaluator(dspy.Module):
    """A DSPy module that runs the translation and Cypher generation steps."""
    def __init__(self):
        super().__init__()
        self.translator = dspy.Predict(Translate)
        self.cypher_generator = dspy.Predict(GenerateCypher)

    def forward(self, question, schema, target_language):
        t = self.translator(source_text=question, target_language=target_language)
        c = self.cypher_generator(question=t.translated_text, schema=schema)
        return dspy.Prediction(translated_question=t.translated_text, generated_query=c.cypher_query)


# This does back translation
def score_translation_quality(tracer, original_text, translated_text, back_translator, sentence_model):
    """Performs back-translation and returns a semantic similarity score."""
    with tracer.start_as_current_span("score_translation_quality") as span:
        back_translation_result = back_translator(source_text=translated_text, target_language="English")
        back_translated_text = back_translation_result.translated_text
        
        embedding_1 = sentence_model.encode(original_text, convert_to_tensor=True)
        embedding_2 = sentence_model.encode(back_translated_text, convert_to_tensor=True)
        
        similarity_score = util.pytorch_cos_sim(embedding_1, embedding_2).item()
        
        span.set_attributes({"score": similarity_score, "back_translated_text": back_translated_text})
        return {"score": similarity_score, "back_translated_text": back_translated_text}


def compare_query_results(tracer, ground_truth_query, generated_query, driver):
    """Executes both queries with granular tracing and compares their results."""
    with tracer.start_as_current_span("compare_query_results"):
        if not driver:
            return {"correct": None, "reason": "Skipped"}
        gt_results, gen_results, gt_error, gen_error = [], [], None, None
        
        with tracer.start_as_current_span("execute_ground_truth_query") as s:
            s.set_attribute("db.statement", ground_truth_query)
            try:
                with driver.session() as session:
                    gt_results = session.run(ground_truth_query).data()
            except Exception as e:
                gt_error = str(e)
                s.set_attribute("error", True)

        with tracer.start_as_current_span("execute_generated_query") as s:
            s.set_attribute("db.statement", generated_query)
            try:
                with driver.session() as session:
                    gen_results = session.run(generated_query).data()
            except Exception as e:
                gen_error = str(e)
                s.set_attribute("error", True)
            
        if gen_error:
            return {"correct": False, "reason": "Execution Error"}
        if gt_error:
            return {"correct": None, "reason": "Ground Truth Error"}
        
        canonical_gt = sorted([tuple(sorted(d.items())) for d in gt_results])
        canonical_gen = sorted([tuple(sorted(d.items())) for d in gen_results])
        
        if canonical_gt == canonical_gen:
            return {"correct": True, "reason": "Results Match"}
        return {"correct": False, "reason": "Results Mismatch"}


def process_and_evaluate_sample(tracer, sample, evaluator, back_translator,
                                sentence_model, target_language, driver):
    """Main function to process one sample, orchestrating all steps."""
    with tracer.start_as_current_span("process_and_evaluate_sample") as s:
        pipeline_result = evaluator(question=sample['question'], schema=sample['schema'], target_language=target_language)
        
        translation_score_result = score_translation_quality(tracer, sample['question'], pipeline_result.translated_question, back_translator, sentence_model)
        
        cypher_assessment = compare_query_results(tracer, sample['cypher'], pipeline_result.generated_query, driver)
        
        s.set_attributes({"question": sample['question'], "translation_score": translation_score_result['score'], "cypher_assessment_correct": cypher_assessment.get('correct')})
        return pipeline_result, translation_score_result, cypher_assessment


all_results = []
target_languages = ["Hindi", "Tamil", "Telugu"]


for model_config in models_to_evaluate:
    model_name = model_config['name']
    llm_instance = model_config['instance']
    
    print(f"\n--- Evaluating Model: {model_name} ---")

    dspy.settings.configure(lm=llm_instance)
    tracer = trace.get_tracer(f"pipeline.{model_name}")

    evaluator = FullPipelineEvaluator()
    back_translator = dspy.Predict(Translate)

    for i, sample in enumerate(train_samples):
        print(f" Processing sample {i+1}/{len(train_samples)} for model '{model_name}'...")
        pipeline_result, translation_score, cypher_assessment = process_and_evaluate_sample(
            tracer=tracer, sample=sample, evaluator=evaluator,
            back_translator=back_translator, sentence_model=sentence_model,
            target_language=target_language, driver=neo4j_driver
        )
        
        all_results.append({
            "model_name": model_name,
            "question": sample['question'],
            "translation_score": translation_score['score'],
            "is_correct": cypher_assessment.get('correct'),
            "reason": cypher_assessment.get('reason'),
            "language": target_language
        })

if neo4j_driver:
    neo4j_driver.close()

print("\n\n--- Evaluation Complete: Final Report ---")
if not all_results:
    print("No results to analyze. Did you set your API keys correctly?")
else:
    df = pd.DataFrame(all_results)

    summary = df.groupby('model_name').agg(
        cypher_accuracy_pct=('is_correct', lambda x: x.mean() * 100),
        avg_translation_score=('translation_score', 'mean')
    ).round(2).reset_index()
    
    print("Overall Model Performance:")
    print(summary.to_string(index=False))

    # Show examples of failures for detailed analysis
    failures_df = df[df['is_correct'] == False]
    if not failures_df.empty:
        print("\nAnalysis of Failures:")
        print(failures_df[['model_name', 'question', 'reason']].to_string(index=False))

print(f"\n All done. View detailed traces in the Phoenix UI: {session.url}")