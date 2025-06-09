import os
from dotenv import load_dotenv

load_dotenv()

# INCLUSION_CRITERIA = [
#     """
#     Population - Populations at risk of developing an NCD (Non-Communicable Diseases)
#     Included are studies of a real or generalizable human population or subpopulation defined by geographic, demographic or social characteristics (e.g., national population, region, age group, socioeconomic group). Excluded are studies where the population consists exclusively of individuals already diagnosed with an NCD (e.g., diabetes, cancer, cardiovascular disease).
#     """,
#     """
#     Intervention, exposure, or scenario (includes comparator) 
#     Included are studies evaluating potential health impacts of exposures, interventions or policies on NCD (Non-Communicable Diseases) outcomes by simulating "what if" scenarios. These include burden-of-disease studies, comparative risk analysis/assessments, and studies of primary or secondary prevention. Excluded are studies focused exclusively on tertiary prevention or methodological development without assessing specific health impact scenarios. 
#     """,
#     """
#     Outcome - Selected non-communicable diseases (NCDs - Non-Communicable Diseases) or NCD risk factors 
#     Included are studies reporting outcomes related to selected major NCDs (cardiovascular diseases, cancer, diabetes, chronic respiratory diseases, mental health conditions, or neurological conditions, injury and musculoskeletal diseases) or their risk factors. Excluded are studies primarily designed as health economic evaluations (e.g., cost-effectiveness analyses) or those focusing exclusively on non-NCD conditions without NCD outcomes.
#     """,
#     """
#     Study approach - Computational simulation modelling 
#     Included are studies using computational simulation modelling methods as the primary analytical tool, such as system dynamics models, agent-based models, microsimulation, Markov models, or attributable risk models. Excluded are observational studies, prediction models for individual risk, statistical regression models, or trend forecasting models that don't simulate interventions. 
#     """,
#     """
#     Integration of multiple data sources 
#     Studies typically integrate multiple distinct data sources, including empirical data, expert opinion or literature-based estimates of model parameters such as causal risk-outcome relationships or intervention effectiveness. 
#     """,
#     """
#     Model adaptability 
#     The model should be adaptable to incorporate new data, test alternative hypotheses or explore additional scenarios.
#     """
# ]

INCLUSION_CRITERIA = [
    """
    Population
    Studies focusing on population-level outcomes of NCDs (Non-Communicable Diseases) and their risks. Focuses on outcomes for groups of people not individuals.
    """,
    """
    Intervention/Exposure
    Assessing the impact of hypothetical health risk exposures, interventions, policies, screening and diagnostic programs, lifestyle intervention, disease projections on NCD outcomes through theoretical or hypothetical scenarios.
    """,
    """
    Comparison
    Comparing the above interventions to current practices relating to these outcomes and other hypothetical practices that may be considered as alternatives.
    """,
    """
    Outcome
    Outcomes related to the 5 primary NCDs (Non-Communicable Diseases). This includes heart disease, diabetes, cancer (not squamous cell or basal cell carcinoma), chronic respiratory diseases (COPD/asthma), mental health and neurological conditions (anxiety, depression, schizophrenia, bipolar, dementia, parkinson's).
    """,
    """
    Study Design
    Computational simulation modeling studies utilizing the following techniques (and more): agent-based modeling, discrete event simulations, system dynamics modelling, Monte Carlo simulations, markov chains, micro/macrosimulation models, and life table analyses.
    """,
    """
    Integration of Data
    Models integrating multiple data sources (e.g., population demographics, disease prevalence, intervention effectiveness).
    """,
    """
    Adaptability
    Models that are flexible and allow for updates, new data integration, and testing of various scenarios.
    """
]

EXCLUSION_CRITERIA = [
    """
    Population
    Studies focusing on individual-level predictions, such as TRIPOD-AI, or clinical trial participants.
    """,
    """
    Intervention/Exposure
    Observational studies without simulation, or those reporting observed outcomes without examining hypothetical scenarios.
    """,
    """
    Comparison
    Studies not involving comparative scenarios or static analyses lacking adaptability to multiple interventions.
    """,
    """
    Outcome
    Studies focusing exclusively on infectious diseases unless explicitly linked as risk factors for NCDs.
    """,
    """
    Study Design
    Studies using only descriptive statistics, regression analyses without simulation, or simple extrapolations.
    """,
    """
    Integration of Data
    Studies relying on a single data source or using hypothetical data without real-world inputs.
    """,
    """
    Adaptability
    Static models or those designed for a single, unchangeable scenario.
    """
]



# CRITERIA_COLORS = {
#     0: (1, 0.9, 0),      # Yellow
#     1: (0.6, 1, 0.6),    # Light green
#     2: (0.8, 0.9, 1),    # Light blue
#     3: (1, 0.6, 0.6),    # Light red
#     4: (0.9, 0.9, 0.6),  # Light yellow
#     5: (0.6, 0.6, 1),    # Light purple
# }

CRITERIA_COLORS = {
    0: (1, 0.9, 0),      # Yellow
    1: (0.6, 1, 0.6),    # Light green
    2: (0.8, 0.9, 1),    # Light blue
    3: (1, 0.6, 0.6),    # Light red
    4: (1.0, 0.8, 0.6),  # Light orange
    5: (0.9, 0.9, 0.6),  # Light yellow
    6: (0.6, 0.6, 1),    # Light purple
}

# CRITERIA_LABELS = {
#     0: "Population",
#     1: "Intervention",
#     2: "Outcome",
#     3: "Study approach",
#     4: "Integration of multiple data sources",
#     5: "Model adaptability"
# }

CRITERIA_LABELS = {
    0: "Population",
    1: "Intervention",
    2: "Comparison",
    3: "Outcome",
    4: "Study Design",
    5: "Integration of Data",
    6: "Adaptability"
}

OPENAI_MODEL = "text-embedding-3-large"
LLM_MODEL = "gpt-4.1-mini"
# CHUNK_SIZE = 100
# OVERLAP = 10
SENTENCES_PER_CHUNK = 4
SENTENCES_OVERLAP = 1
# SIMILARITY_THRESHOLD = 0.5
# SIMILARITY_THRESHOLDS = [0.4, 0.45, 0.42, 0.48, 0.37, 0.37]
SIMILARITY_THRESHOLDS = [0.05, 0.05, 0.05, 0.05, 0.05, 0.13, 0.1]
PDF_FOLDER = "data/papers"
OUTPUT_FOLDER = "data/output"
PLOT_FOLDER = "data/plots"
OVERWRITE = True
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
