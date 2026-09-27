"""Launch the local application at http://127.0.0.1:8765."""
import uvicorn

if __name__ == '__main__':
    uvicorn.run('climb_app.server:app', host='127.0.0.1', port=8765)
