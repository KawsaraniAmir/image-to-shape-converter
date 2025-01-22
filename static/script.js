document.getElementById('upload-form').addEventListener('submit', async (e) => {
    e.preventDefault();

    const formData = new FormData();
    formData.append('image1', document.getElementById('image1').files[0]);
    formData.append('image2', document.getElementById('image2').files[0]);

    const response = await fetch('/compare', {
        method: 'POST',
        body: formData
    });

    const result = await response.json();
    if (response.ok) {
        document.getElementById('result').textContent = `Similarity: ${result.similarity.toFixed(2)}%`;
    } else {
        document.getElementById('result').textContent = `Error: ${result.error}`;
    }
});
