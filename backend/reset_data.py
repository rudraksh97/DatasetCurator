import asyncio
import os
import boto3
from sqlalchemy import text
from db import get_engine

async def reset_db():
    print("Resetting database...")
    try:
        engine = get_engine()
        async with engine.begin() as conn:
            # Terminate other connections to allow DROP
            await conn.execute(text("""
                SELECT pg_terminate_backend(pid) 
                FROM pg_stat_activity 
                WHERE datname = current_database() 
                AND pid <> pg_backend_pid()
            """))
            await conn.execute(text("DROP SCHEMA public CASCADE"))
            await conn.execute(text("CREATE SCHEMA public"))
            await conn.execute(text("GRANT ALL ON SCHEMA public TO public"))
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        print("Database reset complete.")
    except Exception as e:
        print(f"Database reset failed: {e}")

def reset_s3():
    bucket_name = os.environ.get("S3_BUCKET")
    if not bucket_name:
        print("S3_BUCKET not set, skipping S3 cleanup.")
        return
        
    print(f"Clearing S3 bucket: {bucket_name}...")
    try:
        s3 = boto3.resource('s3')
        bucket = s3.Bucket(bucket_name)
        bucket.objects.all().delete()
        print("S3 bucket cleared.")
    except Exception as e:
        print(f"Error clearing S3: {e}")

async def main():
    reset_s3()
    await reset_db()

if __name__ == "__main__":
    asyncio.run(main())
