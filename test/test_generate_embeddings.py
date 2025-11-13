import pytest
import os
import json
import pandas as pd
import numpy as np

from generate_embeddings import generate_embeddings, process_json_files, generate_csvs

MOCK_JSON_DATA = {
    "ciphertext": "A B C A B C D",
    "plaintext": "ABCABCD",
    "key": {
        "a": ["A"],
        "b": ["B"],
        "c": ["C"],
        "d": ["D"]
    }
}

@pytest.fixture
def mock_generate_embeddings(mocker):
    """
    Mock the generate_embeddings function to return predictable data.
    """
    # Create predictable mock embeddings (2 symbols, 2 dimensions)
    mock_embeddings = np.array([[0.1, 0.2], [0.3, 0.4]])
    mock_vocab = ['A', 'B']
    mock_vocab_plaintext = ['A', 'B', 'C']
    
    def fake_generate_embeddings(cipher_json, plaintext=False, **kwargs):
        if plaintext:
            # When plaintext=True, use a different vocab size for distinction
            return mock_embeddings, mock_vocab_plaintext, {'A': 0, 'B': 1, 'C': 2}
        else:
            return mock_embeddings, mock_vocab, {'A': 0, 'B': 1}
    
    # Apply the mock globally
    return mocker.patch(
        'generate_embeddings.generate_embeddings',  # Replace with your actual module path
        side_effect=fake_generate_embeddings
    )


@pytest.fixture
def mock_os(mocker):
    """
    Mock os.listdir and os.makedirs to control file system behavior.
    """
    # Create mock directory structure (tmpdir handles this)
    
    # Mock os.listdir to return our specific JSON files
    mock_listdir = mocker.patch('os.listdir')
    
    # Set up files that will be used by generate_csvs
    mock_listdir.return_value = [
        'cipher-1.json', 
        'cipher-2.json', 
        'test-cipher-A.json', 
        'not_a_cipher.txt'
    ]
    
    # *** REMOVE or COMMENT OUT THIS LINE THAT CAUSES RECURSION: ***
    # mocker.patch('os.path.join', side_effect=lambda *args: os.path.join(*args)) 
    
    # Just mock os.makedirs since it's called inside the function
    mocker.patch('os.makedirs') 
    
    return {
        'listdir': mock_listdir
        # No need to return the tmpdir paths unless used by the test directly
    }

@pytest.fixture
def mock_file_io(mocker):
    """
    Mock file opening and json loading.
    """
    # Mock built-in 'open' for reading the JSON file
    mock_open = mocker.mock_open(read_data=json.dumps(MOCK_JSON_DATA))
    mocker.patch('builtins.open', mock_open)
    
    # Mock json.load to return the predictable data
    mocker.patch('json.load', return_value=MOCK_JSON_DATA)
    
    # Mock pandas to_csv method to check calls instead of writing files
    mock_to_csv = mocker.patch('pandas.DataFrame.to_csv')
    
    return {'open': mock_open, 'to_csv': mock_to_csv}

# --- Tests for generate_embeddings (Core Logic) ---

def test_generate_embeddings_ciphertext():
    """Test co-occurrence and PPMI generation for ciphertext."""
    # Use small window and embedding size for predictable results
    embeddings, vocab, _ = generate_embeddings(
        MOCK_JSON_DATA, 
        window=1, 
        embedding_size=2
    )
    
    # Expected vocab based on MOCK_JSON_DATA["ciphertext"]
    assert sorted(vocab) == ['A', 'B', 'C', 'D']
    assert embeddings.shape == (4, 2)
    # The actual PPMI/SVD values are complex to check directly, 
    # so we primarily check shape and non-zero-ness for sanity.
    assert not np.all(embeddings == 0)

def test_generate_embeddings_plaintext():
    """Test co-occurrence and PPMI generation for plaintext."""
    embeddings, vocab, _ = generate_embeddings(
        MOCK_JSON_DATA, 
        plaintext=True, 
        window=1, 
        embedding_size=2
    )
    
    # Expected vocab based on MOCK_JSON_DATA["plaintext"]
    assert sorted(vocab) == ['A', 'B', 'C', 'D']
    assert embeddings.shape == (4, 2)
    assert not np.all(embeddings == 0)

# --- Tests for process_json_files (Helper Logic) ---

def test_process_json_files_saves_correct_csvs(mock_file_io):
    """Test that process_json_files correctly saves two CSVs per file."""
    
    json_list = ['cipher-1.json', 'cipher-2.json']
    output_dir = 'ciphers_train'
    input_dir = 'ciphers'
    
    # Call the function
    process_json_files(input_dir, json_list, output_dir, 'Training')
    
    # Check that open was called twice (once for each JSON)
    assert mock_file_io['open'].call_count == 2
    
    # Check that to_csv was called four times (2 JSON files * 2 CSVs each)
    assert mock_file_io['to_csv'].call_count == 4
    
    # Check the filenames passed to to_csv (last call is the last file's mappings)
    # Note: We use call_args[0][0] to get the first argument (the path string)
    assert mock_file_io['to_csv'].call_args_list[3][0][0] == os.path.join(
        output_dir, 'cipher-2_mappings.csv'
    )
    assert mock_file_io['to_csv'].call_args_list[2][0][0] == os.path.join(
        output_dir, 'cipher-2_embeddings.csv'
    )

# --- Tests for generate_csvs (Main Logic) ---

def test_generate_csvs_separates_files_and_calls_process(mocker, mock_os):
    """Test that generate_csvs correctly splits files and calls process_json_files."""
    
    # Mock the helper function to check its calls
    mock_process = mocker.patch('generate_embeddings.process_json_files')
    
    # Call the main function
    generate_csvs()
    
    # Check that process_json_files was called twice (once for train, once for test)
    assert mock_process.call_count == 2
    
    # Check the call for training files (starts with 'cipher-')
    train_call = mock_process.call_args_list[0]
    assert train_call[0][2] == 'ciphers_train'
    assert sorted(train_call[0][1]) == ['cipher-1.json', 'cipher-2.json']

    # Check the call for testing files (starts with 'test-cipher-')
    test_call = mock_process.call_args_list[1]
    assert test_call[0][2] == 'ciphers_test'
    assert test_call[0][1] == ['test-cipher-A.json']

def test_generate_csvs_no_files(mocker, mock_os):
    """Test the edge case where no JSON files are found."""
    
    # Make os.listdir return an empty list
    mock_os['listdir'].return_value = ['not_a_file.txt']
    
    # Mock the helper function to ensure it's not called
    mock_process = mocker.patch('generate_embeddings.process_json_files')
    
    # Mock print to capture output
    mock_print = mocker.patch('builtins.print')
    
    generate_csvs()
    
    # Check that no files were processed
    mock_process.assert_not_called()
    mock_print.assert_called_with("No JSON files found in the input directory 'ciphers'")