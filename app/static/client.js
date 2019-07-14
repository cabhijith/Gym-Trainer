var el = x => document.getElementById(x);

function showPicker() {
  el("file-input").click();
}

function showPicked(input) {
  el("upload-label").innerHTML = input.files[0].name;
  var reader = new FileReader();
  reader.onload = function(e) {
    el("image-picked").src = e.target.result;
    el("image-picked").className = "";
  };
  reader.readAsDataURL(input.files[0]);
}

function analyze() {
  var uploadFiles = el("file-input").files;
  if (uploadFiles.length !== 1) alert("Please select a file to analyze!");

  el("analyze-button").innerHTML = "Finding out...";
  var xhr = new XMLHttpRequest();
  var loc = window.location;
  xhr.open("POST", `${loc.protocol}//${loc.hostname}:${loc.port}/analyze`,
    true);
  xhr.onerror = function() {
    alert(xhr.responseText);
  };
  xhr.onload = function(e) {
    if (this.readyState === 4) {
      var response = JSON.parse(e.target.responseText);
      el("title-label").innerHTML = `${response["title"]}`;
      el("info-label").innerHTML = `${response["info"]}`;
      el("ques1-label").innerHTML = `How often should you use it?`;
      el("often-label").innerHTML = `${response["often"]}`;
      el("ques2-label").innerHTML = `Muscles Worked:`;
      el("muscles-label").innerHTML = `${response["muscles"]}`;
      el("ques3-label").innerHTML = `More Tips`;
      el("tips-label").innerHTML = `${response["tips"]}`;




     
      
      
    }
    el("analyze-button").innerHTML = "Analyze";
  };

  var fileData = new FormData();
  fileData.append("file", uploadFiles[0]);
  xhr.send(fileData);
}

