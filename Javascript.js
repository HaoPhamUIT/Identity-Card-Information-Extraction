// script.js
const box = document.getElementById("clickable-box");
const input = document.querySelector("input[type='file']");
box.addEventListener("click", () => {
    input.click();
});
