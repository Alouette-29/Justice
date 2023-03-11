$("#button").click(function () {
    //点击“点击分析”按钮，使input-file触发点击事件，完成读取文件的操作
    $("#input-file").click();
});

function readFile(){
    var selectedFile = document.getElementById("input-file").files[0];
    var name = selectedFile.name;
    var size = selectedFile.size;
    console.log("文件名:" + name + "大小:" + size);

    var reader = new FileReader();
    reader.readAsText(selectedFile);

    reader.onload = function () {
        const content = this.result;
        // console.log(content);
        const resultString = content.toString();
        console.log(resultString);

        // 创建HTTP请求
        const xhr = new XMLHttpRequest();
        xhr.open('POST', '/result2', true);
        xhr.setRequestHeader('Content-Type', 'text/plain;charset=UTF-8');

        // 将文件内容作为文本内容发送到服务器端
        xhr.send(resultString);

        xhr.onreadystatechange = function () {
            if (xhr.status == 200) {
                console.log("请求成功");
            } else {
                console.log("请求失败");
            }
        }
    };
}


// const fileInput = document.getElementById('input-file');
// fileInput.addEventListener('change', readFile);

// function readFile() {
//     const file = fileInput.files[0];
//     const reader = new FileReader();
//     reader.readAsText(file);

//     reader.onload = function() {
//         const content = reader.result;
//         console.log(content)
//         const resultString = content.toString();
//         console.log(resultString);
    
//         // 创建HTTP请求
//         const xhr = new XMLHttpRequest();
//         xhr.open('POST', '/result', true);
//         xhr.setRequestHeader('Content-Type', 'text/plain;charset=UTF-8');
    
//         // 将文件内容作为文本内容发送到服务器端
//         xhr.send(resultString);
//     };
//   }