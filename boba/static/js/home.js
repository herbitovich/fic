
// Show the custom upload button and handle file selection
const fileInput = document.getElementById('imageUpload');
const fileNameDisplay = document.getElementById('fileName');

const customUploadButton = document.createElement('button');
customUploadButton.className = 'custom-upload';
customUploadButton.innerText = 'Upload Image';
customUploadButton.onclick = () => fileInput.click();

fileInput.parentNode.insertBefore(customUploadButton, fileInput.nextSibling);

fileInput.addEventListener('change', function() {
    if (fileInput.files.length > 0) {
        fileNameDisplay.textContent = fileInput.files[0].name;
    } else {
        fileNameDisplay.textContent = 'No file chosen';
    }
});
document.getElementById('sendRequest').addEventListener('click', function() {
    const fileInput = document.getElementById('imageUpload');
    const file = fileInput.files[0];

    if (!file) {
        alert('Please upload an image first.');
        return;
    }

    const formData = new FormData();
    formData.append('image', file);
    document.getElementById('arrow').setAttribute("src", "/static/img/R.gif");
    document.getElementById('arrow').setAttribute("style", "width: 60px;");
    console.log("changed the icon 1")
    fetch('/', {
        method: 'POST',
        body: formData,
    })
    .then(response => response.json())
    .then(data => {
        // Assuming the server returns the URL of the processed image
        document.getElementById('outputImage').innerHTML = `<img src="" alt="Output Image" style="max-width: 100%; max-height: 100%;">`;
        document.getElementById('outputImage').innerHTML = `<img src="${data.imageUrl}" alt="Output Image" style="max-width: 100%; max-height: 100%;">`;
        document.getElementById('arrow').setAttribute("src", "/static/img/arrows-exchange.svg");
        document.getElementById('arrow').setAttribute("style", "filter: invert(1); width: 60px;");
        console.log("success, changed the icon 2")
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('arrow').setAttribute("src", "/static/img/arrows-exchange.svg");
        document.getElementById('arrow').setAttribute("style", "filter: invert(1); width: 60px;");
        console.log("error, changed the icon 2")
    });
    
});
