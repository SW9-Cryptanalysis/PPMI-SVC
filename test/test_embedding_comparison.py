import pytest
import numpy as np
import pandas as pd
from embedding_comparison import find_k_similar_vecs, read_embeddings, read_mappings

@pytest.fixture
def mock_embeddings_data():
    """Provides mock data for read_embeddings to return."""
    # The actual keys in the dicts need to match the data used for find_k_similar_vecs
    cipher_embs = {
        101: np.array([0.9, 0.1, 0.1]),  # Vowel match 'A' (1.0)
        102: np.array([0.8, 0.2, 0.0]),  # Vowel match 'E' (0.94)
        103: np.array([0.0, 0.9, 0.1]),  # Consonant match 'J' (0.75)
        104: np.array([0.0, 0.1, 0.9]),  # Good match for a different letter
        105: np.array([0.4, 0.4, 0.4]),  # Neutral/Noise
    }
    # Plaintext embeddings for 'a', 'e', 'j'
    plaintext_embs = {
        'a': np.array([1.0, 0.0, 0.0]),
        'e': np.array([0.8, 0.0, 0.0]), # Should be slightly less similar to 101 than 'a'
        'j': np.array([0.0, 1.0, 0.0]),
        'b': np.array([0.0, 0.0, 1.0]),
    }
    return cipher_embs, plaintext_embs

@pytest.fixture
def mock_mappings_data():
    """Provides mock mapping data for read_mappings."""
    return {
        101: 'A',
        102: 'E',
        103: 'J',
        104: 'B',
        105: 'X',
    }


@pytest.fixture
def mock_csv_files(mocker):
    """Mocks pandas.read_csv to return predictable DataFrames."""
    
    # --- Cipher Symbol Embeddings DataFrame ---
    c_data = {
        'Symbol': [101, 102, 103, 104, 105],
        'Dim_1': [0.9, 0.8, 0.0, 0.0, 0.4],
        'Dim_2': [0.1, 0.2, 0.9, 0.1, 0.4],
        'Dim_3': [0.1, 0.0, 0.1, 0.9, 0.4],
    }
    mock_c_df = pd.DataFrame(c_data)

    # --- Plaintext Letter Embeddings DataFrame ---
    p_data = {
        'Letter': ['a', 'e', 'j', 'b'],
        'Dim_1': [1.0, 0.8, 0.0, 0.0],
        'Dim_2': [0.0, 0.0, 1.0, 0.0],
        'Dim_3': [0.0, 0.0, 0.0, 1.0],
    }
    mock_p_df = pd.DataFrame(p_data)
    
    # --- Mappings DataFrame ---
    m_data = {
        'cipher_symbol': [101, 102, 103, 104, 105],
        'plaintext_symbol': ['a', 'e', 'j', 'b', 'x'],
    }
    mock_m_df = pd.DataFrame(m_data)

    def mock_read_csv(filepath):
        # FIX: Explicitly check for the mock file paths used in the test.
        if filepath == 'mock_c.csv': 
            return mock_c_df
        elif filepath == 'mock_p.csv':
            return mock_p_df
        elif 'mappings' in filepath:
            return mock_m_df
        
        # This fallback is now only for unexpected paths
        raise FileNotFoundError(f"Mocking error: Unknown file {filepath}")

    mocker.patch('pandas.read_csv', side_effect=mock_read_csv)
    
    return mock_c_df, mock_p_df, mock_m_df

def test_read_embeddings(mock_csv_files):
    """Test read_embeddings correctly parses mock CSVs into dictionaries."""
    cipher_embs, plaintext_embs = read_embeddings('mock_c.csv', 'mock_p.csv')
    
    # Check return types and number of items
    assert isinstance(cipher_embs, dict)
    assert isinstance(plaintext_embs, dict)
    assert len(cipher_embs) == 5
    assert len(plaintext_embs) == 4
    
    # Check a specific key and vector value
    assert 101 in cipher_embs
    assert np.allclose(cipher_embs[101], np.array([0.9, 0.1, 0.1]))
    assert 'a' in plaintext_embs
    assert np.allclose(plaintext_embs['a'], np.array([1.0, 0.0, 0.0]))
    
    # Check sorting for plaintext_embs ('b' should be after 'a')
    assert list(plaintext_embs.keys()) == ['a', 'b', 'e', 'j']


def test_read_mappings(mock_csv_files):
    """Test read_mappings correctly parses mock CSV into a cipher->plaintext map."""
    mappings = read_mappings('mock_mappings.csv')
    
    assert isinstance(mappings, dict)
    assert len(mappings) == 5
    assert mappings[101] == 'A' # Check for uppercasing
    assert mappings[105] == 'X'


def test_find_k_similar_vecs(mock_embeddings_data, mock_mappings_data, capsys):
    """Test find_k_similar_vecs calculates similarities and sorts correctly."""
    cipher_embs, plaintext_embs = mock_embeddings_data
    mappings = mock_mappings_data
    target_vec_a = plaintext_embs['a'] # [1.0, 0.0, 0.0]
    k = 3
    
    result = find_k_similar_vecs(target_vec_a, cipher_embs, mappings, k)
    
    # Expected similarities (calculated with L2 norm for clarity, but cosine is similar):
    # 101: [0.9, 0.1, 0.1] -> Sim: ~0.9997 (should be highest)
    # 102: [0.8, 0.2, 0.0] -> Sim: ~0.9419 (should be second)
    # 105: [0.4, 0.4, 0.4] -> Sim: ~0.5774 (should be third)
    
    expected_keys = [101, 102, 105]
    
    assert list(result.keys()) == expected_keys
    
    # Check if the highest similarity is close to the expected value (101)
    assert result[101] > result[102]
    assert result[102] > result[105]
    assert result[101] == pytest.approx(0.9 / np.sqrt(0.83), abs=1e-3) # Expected cosine sim
    
    # Check print output for formatting and mapping usage
    captured = capsys.readouterr()
    assert "Match: A (Symbol: 101), Similarity:" in captured.out
    assert "Match: E (Symbol: 102), Similarity:" in captured.out
    assert "Match: X (Symbol: 105), Similarity:" in captured.out