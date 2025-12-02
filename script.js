document.getElementById('image').addEventListener('change', function(e) {
    const preview = document.getElementById('preview');
    if (e.target.files.length > 0) {
        preview.src = URL.createObjectURL(e.target.files[0]);
        preview.classList.remove('d-none');
    } else {
        preview.classList.add('d-none');
    }
});

document.getElementById('upload-form').addEventListener('submit', function(e) {
    const progress = document.getElementById('upload-progress');
    const bar = progress.querySelector('.progress-bar');
    progress.classList.remove('d-none');
    let width = 0;
    const interval = setInterval(() => {
        width += 10;
        bar.style.width = width + '%';
        if (width >= 100) {
            clearInterval(interval);
        }
    }, 80);
});
document.addEventListener("DOMContentLoaded", function() {
    const bars = document.querySelectorAll(".progress-bar");
    bars.forEach(bar => {
        bar.style.transition = "width 1s ease-in-out";
        bar.style.width = bar.getAttribute("aria-valuenow") + "%";
    });
});