# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import click
import pathlib
from aiohttp import web
import aiofiles
import asyncio

# %%
routes = web.RouteTableDef()
jobs = list()

@routes.get('/')
async def root(request):
    raise web.HTTPFound('/job/0')

async def write_file(path, text):
    async with aiofiles.open(path, mode='w') as f:
        await f.write(text)

@routes.get('/job/{job_id}')
async def check(request):
    if 'prev_job' in request.query:
        prev_job_id = int(request.query['prev_job'])
        is_true = request.query['result'] == 'true'
        prev_job = jobs[prev_job_id]
        result_path = prev_job['result_path']
        text = 'true' if is_true else 'false'
        task = asyncio.create_task(write_file(result_path, text))
        prev_job['task'] = task
        prev_job['result'] = is_true
        print(prev_job_id, is_true)
        
    job_id = int(request.match_info['job_id'])
    if job_id >= len(jobs):
        n_results = 0
        n_true = 0
        for job in jobs:
            if 'result' in job:
                n_results += 1
                if job['result']:
                    n_true += 1
        ratio = n_true / n_results * 100
        return web.Response(text=f"All Done! {ratio}% of the videos are kept!")
    
    job = jobs[job_id]
    vid_path = job['vid_path']
    
    return web.Response(
        text=f"""
        <video autoplay loop muted playsinline style="width: 50%"><source src="/video/{vid_path}" type="video/mp4"> </video>
        <br>
        <a href="/job/{job_id-1}" style="font-size:10em;color:inherit;text-decoration:none;">⏮️</a>
        <a href="/job/{job_id+1}?prev_job={job_id}&result=true" style="font-size:10em;color:inherit;text-decoration:none;">✅</a>
        <a href="/job/{job_id+1}?prev_job={job_id}&result=false" style="font-size:10em;color:inherit;text-decoration:none;">❌</a>
        """,
        content_type='text/html')

@click.command()
@click.argument('session_dir', nargs=-1)
def main(session_dir):
    try:
        for session in session_dir:
            session = pathlib.Path(os.path.expanduser(session))
            name = session.name
            routes.static(f'/video/{name}', path=session.absolute())
            
            demos_dir = session.joinpath('demos')
            for vid_path in sorted(demos_dir.glob("demo*/video_thumbnail.mp4")):
                rel_path = vid_path.relative_to(session.parent)
                result_path = vid_path.parent.joinpath('check_result.txt')
                if result_path.is_file():
                    continue
                
                jobs.append({
                    'vid_path': str(rel_path),
                    'result_path': result_path.absolute()
                })
        print(f"{len(jobs)} jobs in total.")
        app = web.Application()
        app.add_routes(routes)
        web.run_app(app)
    except KeyboardInterrupt:
        tasks = list()
        for job in jobs:
            if 'task' in job:
                tasks.append(job['task'])
        
        n_results = 0
        n_true = 0
        for job in jobs:
            if 'result' in job:
                n_results += 1
                if job['result']:
                    n_true += 1
        ratio = n_true / n_results * 100
        print(f"{ratio}% of the videos are kept!")
        asyncio.gather(tasks)
    
# %%
if __name__ == "__main__":
    main()
