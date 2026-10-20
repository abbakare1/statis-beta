"""Persistence test: create account, upload data, analyze, restart, verify."""

import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config.settings import get_settings
from app.core.database import init_db, close_db, get_session
from app.security.auth import create_user, create_access_token, create_invitation, redeem_invitation
from app.models import Project, Dataset

settings = get_settings()

# Clean up any existing database
if settings.DB_PATH.exists():
    settings.DB_PATH.unlink()

async def setup():
    """Initialize database and create test users."""
    await init_db()
    
    # Create admin user
    async with get_session() as session:
        admin = await create_user(session, "admin@test.com", "password123", "Admin User", role="admin")
        researcher = await create_user(session, "researcher@test.com", "password123", "Researcher User", role="researcher")
        await session.commit()
    
    return admin, researcher

async def create_project(user_id, name):
    """Create a project for a user."""
    async with get_session() as session:
        project = Project(
            name=name,
            research_question="Does treatment affect outcome?",
            study_design="cohort",
            owner_id=user_id,
        )
        session.add(project)
        await session.commit()
        await session.refresh(project)
        return project

async def upload_dataset(project_id, filename, content):
    """Upload a dataset."""
    upload_path = settings.UPLOAD_DIR / f"{project_id}_{filename}"
    with open(upload_path, "w") as f:
        f.write(content)
    
    # Compute content hash
    with open(upload_path, "rb") as f:
        content_hash = hashlib.sha256(f.read()).hexdigest()
    
    async with get_session() as session:
        dataset = Dataset(
            project_id=project_id,
            original_filename=filename,
            stored_path=str(upload_path),
            content_hash=content_hash,
            row_count=10,
            column_count=4,
        )
        session.add(dataset)
        await session.commit()
        await session.refresh(dataset)
        return dataset

async def verify_data(user_id, project_id, dataset_id):
    """Verify that data persists."""
    async with get_session() as session:
        # Verify project
        from sqlalchemy import select
        result = await session.execute(select(Project).where(Project.id == project_id))
        project = result.scalar_one_or_none()
        assert project is not None, "Project not found"
        assert project.owner_id == user_id, "Project owner mismatch"
        
        # Verify dataset
        result = await session.execute(select(Dataset).where(Dataset.id == dataset_id))
        dataset = result.scalar_one_or_none()
        assert dataset is not None, "Dataset not found"
        assert dataset.project_id == project_id, "Dataset project mismatch"
        
        # Verify file exists
        assert Path(dataset.stored_path).exists(), "Dataset file not found"
        
        return True

async def main():
    print("=== STATIS Persistence Test ===")
    
    # Setup
    print("\n1. Setting up database and users...")
    admin, researcher = await setup()
    print(f"   Admin user: {admin.id}")
    print(f"   Researcher user: {researcher.id}")
    
    # Create project
    print("\n2. Creating project...")
    project = await create_project(researcher.id, "Persistence Test Project")
    print(f"   Project: {project.id}")
    
    # Upload dataset
    print("\n3. Uploading dataset...")
    csv_content = "patient_id,age,treatment,outcome\n"
    for i in range(10):
        csv_content += f"P{i},50,1,{i*10}\n"
    dataset = await upload_dataset(project.id, "test_data.csv", csv_content)
    print(f"   Dataset: {dataset.id}")
    print(f"   Content hash: {dataset.content_hash}")
    
    # Verify data before restart
    print("\n4. Verifying data before restart...")
    await verify_data(researcher.id, project.id, dataset.id)
    print("   Data verified before restart")
    
    # Shutdown database
    print("\n5. Shutting down database...")
    await close_db()
    
    # Simulate restart by reinitializing database
    print("\n6. Reinitializing database (simulating restart)...")
    await init_db()
    
    # Verify data after restart
    print("\n7. Verifying data after restart...")
    await verify_data(researcher.id, project.id, dataset.id)
    print("   Data verified after restart")
    
    # Cleanup
    print("\n8. Cleaning up...")
    await close_db()
    
    print("\n=== Persistence Test PASSED ===")

if __name__ == "__main__":
    asyncio.run(main())