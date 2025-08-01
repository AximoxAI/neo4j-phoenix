# Multilingual Text2Cypher for Indian Languages: A Cross-Language Family Evaluation Framework

## Abstract

We present a comprehensive study on Text2Cypher generation across six major Indian languages from two distinct language families: Indo-Aryan (Hindi, Bengali, Marathi) and Dravidian (Telugu, Tamil, Kannada). Our work focuses on translating the existing Neo4j Text2Cypher dataset and evaluating smaller, open-source models including Sarvam-M for cross-lingual query generation performance. We establish a novel dual-evaluation framework assessing both translation quality and downstream Text2Cypher generation, using Gemini as our primary LLM-as-Judge. Our pipeline integrates DSPy for robust query execution and result validation, with comprehensive monitoring through Phoenix (Arize AI) for reproducible research and deployment insights.

## 1. Introduction

Graph databases require sophisticated query languages like Cypher, creating accessibility barriers for non-English speakers. With over 1.3 billion speakers across six major Indian languages spanning two distinct language families—Indo-Aryan and Dravidian—there exists a critical need for multilingual Text2Cypher capabilities. This work addresses three key research questions: (1) How does translation quality affect downstream Text2Cypher performance across language families? (2) Can smaller, specialized models like Sarvam-M compete with larger general-purpose models for Indian languages? (3) What are the systematic differences between Indo-Aryan and Dravidian language performance in structured query generation?

Our contributions include: systematic translation and evaluation of the Neo4j Text2Cypher dataset across 6 Indian languages, comparative analysis of language family effects on query generation, comprehensive evaluation of open-source models with focus on Sarvam-M, and dual-evaluation framework for both translation quality and Text2Cypher generation performance.

## 2. Language Family Analysis and Dataset Translation

### 2.1 Target Languages and Linguistic Characteristics

**Indo-Aryan Family (Indo-European):**
- **Hindi** (हिन्दी): 600M+ speakers, Devanagari script, SOV word order
- **Bengali** (বাংলা): 300M+ speakers, Bengali script, complex conjunct consonants  
- **Marathi** (मराठी): 90M+ speakers, Devanagari script, case marking system

**Dravidian Family:**
- **Telugu** (తెలుగు): 95M+ speakers, Telugu script, agglutinative morphology
- **Tamil** (தமிழ்): 80M+ speakers, Tamil script, no inherent vowel
- **Kannada** (ಕನ್ನಡ): 60M+ speakers, Kannada script, complex sandhi rules

### 2.2 Translation Pipeline Architecture

Rather than creating new datasets, we focus on systematic translation of the existing `neo4j/text2cypher-2025v1` dataset, preserving the rich schema information and query complexity while adapting natural language components for Indian linguistic contexts.

**Translation Models Evaluated:**
- **Sarvam-M**: Open-source model specialized for Indian languages
- **Gemma-2-9B**: Google's efficient multilingual model
- **Llama-3.1-8B**: Meta's open-source multilingual architecture
- **Qwen2.5-7B**: Alibaba's multilingual model with Indian language support

**Translation Strategy:**
- Preserve all Cypher syntax, keywords, and schema elements in English
- Translate natural language queries while maintaining semantic equivalence
- Apply language family-specific linguistic adaptations
- Implement script-aware tokenization for complex writing systems

### 2.3 Language Family-Specific Considerations

**Indo-Aryan Language Adaptations:**
- Handle complex case marking systems (Hindi/Marathi)
- Manage conjunct consonant combinations (Bengali)
- Preserve honorific levels and formality markers
- Address compound word structures in Devanagari

**Dravidian Language Adaptations:**
- Handle agglutinative morphology and suffix combinations
- Manage complex vowel-consonant ligatures
- Preserve aspectual and temporal markers
- Address language-specific phonological rules

## 3. Dual Evaluation Framework

### 3.1 Translation Quality Assessment

We implement comprehensive translation evaluation using Gemini as our primary LLM-as-Judge to assess translation quality before downstream task evaluation.

**Translation Evaluation Dimensions:**
- **Semantic Preservation**: Maintaining original query intent across languages
- **Linguistic Fluency**: Natural language quality in target languages
- **Technical Term Preservation**: Ensuring graph database terminology remains intact
- **Cultural Appropriateness**: Adapting examples for Indian business contexts

**Gemini-as-Judge Implementation:**
```python
# Translation quality assessment prompt
translation_eval_prompt = """
Evaluate the translation quality from English to {target_language}:

Original: {english_text}
Translation: {translated_text}

Assess on 1-5 scale:
1. Semantic equivalence
2. Linguistic fluency  
3. Technical term preservation
4. Cultural appropriateness

Provide scores and brief justification.
"""
```

### 3.2 Text2Cypher Generation Evaluation

Following translation assessment, we evaluate downstream Text2Cypher generation performance using the translated queries as input to various models.

**Text2Cypher Models Evaluated:**
- **Sarvam-M**: Primary focus on Indian language specialization
- **Gemma-2-9B**: Baseline multilingual performance
- **Llama-3.1-8B**: Open-source alternative comparison
- **Qwen2.5-7B**: Multilingual architecture with broad language support

**Generation Evaluation Protocol:**
- Few-shot prompting with language-specific examples
- Schema-aware query generation with context preservation
- Cross-language family performance comparison
- Error analysis by linguistic and structural factors

### 3.3 DSPy Integration for Query Execution

We leverage DSPy for robust query execution and result validation, enabling systematic evaluation of generated Cypher queries.

**DSPy Pipeline Implementation:**
```python
import dspy
from dspy import ChainOfThought, Predict

class CypherGenerator(dspy.Signature):
    """Generate Cypher query from natural language question"""
    question = dspy.InputField(desc="Natural language question")
    schema = dspy.InputField(desc="Graph database schema")
    cypher = dspy.OutputField(desc="Valid Cypher query")

class CypherValidator(dspy.Signature):
    """Validate Cypher query execution and results"""
    cypher = dspy.InputField(desc="Generated Cypher query")
    expected_result = dspy.InputField(desc="Expected query result")
    validation = dspy.OutputField(desc="Execution success and result accuracy")

# DSPy modules for systematic evaluation
cypher_gen = ChainOfThought(CypherGenerator)
cypher_val = Predict(CypherValidator)
```

### 3.4 Gemini-as-Judge for Cypher Quality

Gemini serves as our comprehensive judge for generated Cypher query quality assessment.

**Cypher Evaluation Dimensions:**
- **Syntactic Correctness**: Valid Cypher syntax and structure validation
- **Semantic Equivalence**: Query intent preservation across languages
- **Schema Compliance**: Proper use of nodes, relationships, and properties
- **Executability**: Successful query execution on target databases
- **Result Accuracy**: Correctness of query outputs compared to expected results

**Gemini Judge Implementation:**
```python
cypher_eval_prompt = """
Evaluate the generated Cypher query:

Question ({language}): {question}
Schema: {schema}
Generated Cypher: {cypher}
Expected Result: {expected_result}

Assess on 1-5 scale:
1. Syntactic correctness
2. Semantic equivalence  
3. Schema compliance
4. Executability
5. Result accuracy

Provide detailed analysis and scores.
"""
```

## 4. Phoenix Integration for Comprehensive Monitoring

### 4.1 Translation Pipeline Tracing

Phoenix provides end-to-end observability for our translation pipeline, enabling detailed analysis of model performance across language families.

**Translation Monitoring Components:**
- **Model Performance Tracking**: Latency and quality metrics per translation model
- **Language Family Analysis**: Comparative performance between Indo-Aryan and Dravidian
- **Error Pattern Detection**: Systematic identification of translation failure modes
- **Quality Score Distribution**: Statistical analysis of Gemini judge assessments

### 4.2 Text2Cypher Generation Monitoring

Comprehensive tracking of query generation performance across all models and languages.

**Generation Monitoring Features:**
- **Cross-Model Comparison**: Sarvam-M vs. other open-source alternatives
- **Language Family Effects**: Performance differences between Indo-Aryan and Dravidian
- **Query Complexity Analysis**: Success rates by query structure and schema complexity
- **Failure Mode Categorization**: Systematic error analysis and pattern recognition

### 4.3 DSPy Execution Tracing

Phoenix integration with DSPy enables detailed monitoring of query execution and validation processes.

**Execution Monitoring Capabilities:**
- **Query Execution Success Rates**: Real-time tracking across languages and models
- **Result Validation Accuracy**: Automated assessment of query output correctness
- **Performance Bottleneck Identification**: Latency analysis and optimization opportunities
- **Schema Utilization Patterns**: Analysis of how models interact with different schemas

### 4.4 Gemini Judge Analytics

Specialized monitoring for evaluation consistency and bias detection in our LLM-as-Judge framework.

**Judge Performance Tracking:**
- **Evaluation Consistency**: Score distribution analysis across languages
- **Language Family Bias Detection**: Systematic preference identification
- **Confidence Calibration**: Correlation between judge confidence and actual accuracy
- **Inter-Evaluation Agreement**: Consistency across different evaluation dimensions

## 5. Experimental Design

### 5.1 Translation Evaluation Protocol

**Phase 1: Multi-Model Translation**
- Translate Neo4j dataset using all four translation models
- Generate translations for all 6 target languages
- Apply language family-specific post-processing
- Collect translation quality metrics using Gemini-as-Judge

**Translation Quality Metrics:**
- BLEU scores for lexical similarity
- Semantic similarity using multilingual embeddings
- Gemini judge scores across four evaluation dimensions
- Human evaluation subset for calibration

### 5.2 Text2Cypher Generation Protocol

**Phase 2: Cross-Model Query Generation**
- Use best translations as input for Text2Cypher generation
- Evaluate all four models (focus on Sarvam-M performance)
- Apply consistent few-shot prompting strategies
- Generate queries across varying complexity levels

**Generation Quality Metrics:**
- ROUGE-L scores for query similarity
- ExactMatch accuracy for perfect query reproduction
- DSPy execution success rates
- Gemini judge comprehensive assessment

### 5.3 Language Family Comparative Analysis

**Cross-Family Performance Study:**
- Statistical comparison between Indo-Aryan and Dravidian performance
- Error analysis by linguistic features (morphology, syntax, script complexity)
- Resource level impact assessment (high vs. medium resource languages)
- Cultural context preservation evaluation

## 6. Expected Research Contributions

### 6.1 Empirical Findings

**Language Family Effects:**
- Systematic analysis of Indo-Aryan vs. Dravidian performance differences
- Impact of morphological complexity on query generation accuracy
- Script complexity correlation with translation and generation quality
- Resource level effects within and across language families

**Model Performance Insights:**
- Sarvam-M specialization benefits vs. general multilingual models
- Open-source model capabilities for Indian languages
- Translation quality impact on downstream task performance
- Optimal model combinations for translation and generation pipelines

### 6.2 Technical Contributions

**Evaluation Framework Innovation:**
- Dual-evaluation methodology for translation and generation tasks
- Comprehensive LLM-as-Judge implementation with bias detection
- DSPy integration for systematic query execution and validation
- Phoenix-powered observability for multilingual NLP research

**Language-Specific Insights:**
- Linguistic feature impact analysis on structured query generation
- Cultural context adaptation strategies for Indian languages
- Error pattern taxonomies by language family and script type
- Best practices for Indian language Text2Cypher development

### 6.3 Open Science Contributions

**Dataset and Code Release:**
- Translated Neo4j dataset across 6 Indian languages
- Comprehensive evaluation annotations with Gemini judge scores
- Complete experimental pipeline with Phoenix monitoring integration
- Reproducible benchmarking suite for future research

## 7. Evaluation Metrics and Analysis

### 7.1 Translation Quality Metrics

**Automatic Metrics:**
- BLEU, ROUGE-L for lexical similarity assessment
- Semantic similarity using multilingual sentence embeddings
- Character-level accuracy for script preservation
- Cultural term preservation accuracy

**LLM-as-Judge Metrics:**
- Gemini assessment scores across four dimensions
- Inter-judge agreement analysis for consistency validation
- Confidence calibration studies
- Bias detection across language families

### 7.2 Text2Cypher Generation Metrics

**Task-Specific Metrics:**
- ExactMatch accuracy for perfect query reproduction
- DSPy execution success rates across different schemas
- Result set accuracy compared to ground truth outputs
- Query complexity handling assessment

**Cross-Lingual Consistency:**
- Semantic equivalence preservation across languages
- Schema utilization consistency
- Error pattern correlation analysis
- Performance degradation measurement

### 7.3 Language Family Analysis

**Comparative Studies:**
- Statistical significance testing between Indo-Aryan and Dravidian performance
- Morphological complexity correlation analysis
- Script complexity impact assessment
- Resource availability effect quantification

## 8. Phoenix Monitoring and Analytics

### 8.1 Real-Time Performance Dashboards

**Translation Pipeline Monitoring:**
- Model latency and throughput metrics
- Quality score distributions by language and model
- Error rate tracking and pattern identification
- Resource utilization optimization insights

**Generation Pipeline Analytics:**
- Cross-model performance comparison visualizations
- Language family performance heat maps
- Query complexity success rate analysis
- Schema-specific performance breakdowns

### 8.2 Advanced Analytics and Insights

**Embedding Analysis:**
- Clustering of translation patterns by linguistic features
- Semantic similarity mapping across language families
- Error pattern embedding visualization
- Model behavior similarity analysis

**Drift Detection:**
- Performance consistency monitoring over time
- Model degradation early warning systems
- Data distribution shift detection
- Quality metric stability analysis

## 9. Reproducibility and Future Work

### 9.1 Open Source Infrastructure

All components of our experimental pipeline will be released as open-source tools:

**Core Components:**
- Multi-model translation pipeline with language family adaptations
- Dual-evaluation framework with Gemini-as-Judge integration
- DSPy-powered query execution and validation system
- Phoenix monitoring configuration for multilingual research

### 9.2 Future Research Directions

**Language Expansion:**
- Extension to additional Indian languages (Gujarati, Punjabi, Odia, Malayalam)
- Code-mixed query handling (Hinglish, Tanglish patterns)
- Regional dialect support within major languages
- Cross-language transfer learning techniques

**Technical Enhancements:**
- Fine-tuning strategies for Indian language specialization
- Cultural context embedding learning
- Multi-modal schema understanding
- Federated learning approaches for privacy-preserving improvement

## Conclusion

This work establishes a comprehensive framework for evaluating multilingual Text2Cypher generation across Indian languages, with a systematic comparison between Indo-Aryan and Dravidian language families. By focusing on translation quality assessment and downstream task performance using smaller, open-source models, such as Sarvam-M, we provide practical insights for deploying multilingual graph database interfaces. Our dual-evaluation methodology, powered by Gemini-as-Judge and DSPy execution validation, combined with Phoenix monitoring, creates a reproducible research infrastructure that advances both scientific understanding and practical accessibility of graph databases for Indian language speakers.# neo4j-phoenix
