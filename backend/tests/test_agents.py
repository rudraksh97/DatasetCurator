from pathlib import Path

from agents.cleaning_agent import CleaningAgent
from agents.ingestion import IngestionAgent
from agents.quality_agent import DataQualityAgent
from agents.schema_agent import SchemaUnderstandingAgent
from agents.versioning_agent import VersioningAgent
from models.dataset_state import DatasetState


DATA_PATH = Path(__file__).resolve().parent.parent / "Data.csv"


def test_ingestion_schema_quality(tmp_path):
    ingestion = IngestionAgent(raw_storage=tmp_path / "raw")
    schema_agent = SchemaUnderstandingAgent()
    quality_agent = DataQualityAgent()

    state = ingestion.ingest_uploaded(DATA_PATH, dataset_id="unit1")
    assert state.raw_path and state.raw_path.exists()

    state = schema_agent.analyze(state)
    assert state.schema is not None

    state = quality_agent.analyze(state)
    assert isinstance(state.quality_issues, list)


def test_cleaning_versioning(tmp_path):
    ingestion = IngestionAgent(raw_storage=tmp_path / "raw")
    cleaning = CleaningAgent(curated_storage=tmp_path / "curated")
    versioning = VersioningAgent(metadata_dir=tmp_path / "meta")

    state = ingestion.ingest_uploaded(DATA_PATH, dataset_id="unit2")
    state = cleaning.apply_fixes(state, approved_fixes=[])
    assert state.curated_path and state.curated_path.exists()

    state = versioning.create_version(state)
    assert state.current_version == 1
    assert state.transformation_log
