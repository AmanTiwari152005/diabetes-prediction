document.getElementById('predictionForm').addEventListener('submit', function (e) {
  e.preventDefault();

  const form = e.target;
  const formData = new FormData(form);
  const data = {};

  formData.forEach((value, key) => {
    data[key] = parseFloat(value);
  });

  fetch('http://127.0.0.1:5000/predict', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(data)
  })
    .then(res => res.json())
    .then(data => {
      document.getElementById('result').innerText = data.result;
    })
    .catch(err => {
      document.getElementById('result').innerText = 'Error occurred while predicting.';
      console.error(err);
    });
});
