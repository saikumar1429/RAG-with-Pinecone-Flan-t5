from unittest.mock import MagicMock
from app import generate_answer

def test_generate_answer():
    mock_index = MagicMock()
    mock_embedder = MagicMock()
    mock_generator = MagicMock()
    
    # Setup mocks
    # embedder.encode(query).tolist()
    mock_embedder.encode.return_value.tolist.return_value = [0.1, 0.2]
    
    # index.query(...)
    mock_index.query.return_value = {"matches": [{"metadata": {"text": "mock context"}}]}
    
    # generator(...)
    mock_generator.return_value = [{"generated_text": "mock answer"}]
    
    query = "test query"
    
    print("Testing generate_answer with corrected argument order...")
    # The fixed signature is generate_answer(query, index, embedder, generator)
    # The corrected call in main is also generate_answer(query, index, embedder, generator)
    # Wait, let's double check app.py definition
    # Line 22: def generate_answer(query,index,embedder,generator):
    # So we must pass (query, index, embedder, generator)
    
    try:
        ans, ctx = generate_answer(query, mock_index, mock_embedder, mock_generator)
        print("Successfully executed generate_answer.")
        print(f"Answer: {ans}")
        print(f"Context: {ctx}")
    except Exception as e:
        print(f"FAILED: {e}")
        raise e

if __name__ == "__main__":
    test_generate_answer()
