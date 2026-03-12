"""Tests for clustering."""
def test_spectral_communities(sample_similarity_matrix):
    from src.clustering.community import spectral_communities
    labels = [f"ASSET_{i}" for i in range(sample_similarity_matrix.shape[0])]
    result = spectral_communities(sample_similarity_matrix, labels)
    assert len(result) == len(labels)
