import pytest
import numpy as np
from plot_embeddings import (plot_embeddings_2d, plot_embeddings_2d_vc)

@pytest.fixture
def mock_data():
    """Provides mock data suitable for plotting and clustering (min 5 points for TSNE)."""
    embeddings = np.array([
        [0.1, 0.2, 0.3],
        [1.1, 1.2, 1.3],
        [2.1, 2.2, 2.3],
        [0.15, 0.25, 0.35],
        [5.0, 5.0, 5.0]
    ])
    vocab = ['1', '2', '3', '4', '5']
    cipher_to_letter = {'1': 'a', '2': 'b', '3': 'c', '4': 'd', '5': 'x'}
    return embeddings, vocab, cipher_to_letter


@pytest.fixture(autouse=True)
def mock_plot_dependencies(mocker):
    """Mocks heavy dependencies like TSNE and Matplotlib functions."""
    
    mock_tsne_instance = mocker.MagicMock()
    mock_tsne_instance.fit_transform.return_value = np.array([
        [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]
    ])
    mocker.patch('sklearn.manifold.TSNE', return_value=mock_tsne_instance)

    mock_scatter_return = mocker.MagicMock()
    mock_scatter_return.legend_elements.return_value = (
        [mocker.MagicMock()],  # Handles
        ['mock_label']         # Labels
    )
    mocker.patch('matplotlib.pyplot.figure')
    mock_savefig = mocker.patch('matplotlib.pyplot.savefig')
    mock_show = mocker.patch('matplotlib.pyplot.show')
    mocker.patch('matplotlib.pyplot.scatter')
    mocker.patch('matplotlib.pyplot.text')
    mocker.patch('matplotlib.pyplot.title')
    mocker.patch('matplotlib.pyplot.legend')
    mocker.patch('matplotlib.pyplot.tight_layout')

    return mock_savefig, mock_show

def test_plot_2d_saves_when_save_path_given(mock_data, mock_plot_dependencies):
    """Test plot_embeddings_2d saves the file and does not show the plot when save_path is provided."""
    embeddings, vocab, ctl = mock_data
    mock_savefig, mock_show = mock_plot_dependencies
    
    save_path = "test_simple.png"
    
    plot_embeddings_2d(embeddings, vocab, save_path=save_path)
    
    mock_savefig.assert_called_once_with(save_path, dpi=150)
    mock_show.assert_not_called()

def test_plot_2d_shows_when_no_save_path(mock_data, mock_plot_dependencies):
    """Test plot_embeddings_2d shows the plot when save_path is None."""
    embeddings, vocab, _ = mock_data
    mock_savefig, mock_show = mock_plot_dependencies
    
    plot_embeddings_2d(embeddings, vocab)
    
    mock_savefig.assert_not_called()
    mock_show.assert_called_once()

def test_plot_2d_vc_coloring_logic(mock_data, mock_plot_dependencies, mocker):
    """Test plot_embeddings_2d_vc uses the vowel/consonant logic."""
    embeddings, vocab, ctl = mock_data
    
    # Need to mock scatter again locally to inspect arguments
    mock_scatter = mocker.patch('matplotlib.pyplot.scatter')
    
    plot_embeddings_2d_vc(embeddings, vocab, cipher_to_letter=ctl)
    
    # Expected colors based on mock_data: A(a) -> red, B(b) -> blue, C(c) -> blue, D(d) -> blue, E(x) -> blue
    # Note: 'x' is not a vowel, so it should be blue.
    expected_colors = ['red', 'blue', 'blue', 'blue', 'blue']
    
    # Check that scatter was called with the correct color array (c=colors)
    _, call_kwargs = mock_scatter.call_args
    assert np.array_equal(call_kwargs['c'], expected_colors)