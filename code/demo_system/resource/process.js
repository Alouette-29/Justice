let fileinput = document.getElementById("file")
//function 
//这是一个读入图片的函数 
let preview = document.getElementById("image-preview")
function replace_preview(img)
{
    preview.src = img
    let imgae = new Image()
    imgae.src= img
    imgae.onload = function(){
        let w = this.width
        let h = this.height
        let rate = 1
        let nw = 600
        let nh = 300
        if(w>nw||h>nh)
        {
            rate = Math.max((w/nw),(h/nh))
            w = Math.ceil(w/rate)
            h = Math.ceil(h/rate)
        }
        preview.width = w
        preview.height = h
    }
}
//增加回调函数到组件中去
//这个事件 监听器 在监听到change 的时候 打开文件 阅读文件，并且去调用replace_preview 

fileinput.addEventListener('change',function(){
    preview.style.background = ''
    if(!fileinput.value)
    {
        return
    }
    let file = fileinput.files[0]
    fileinput = file
    let reader = new FileReader()
    reader.onload = function(e)
    {
        let data = e.target.result
        //replace_preview(data)
        console.log(data)
        console.log('打印点人话')
    }
    reader.readAsDataURL(file)
})

// function entity_recongnition()
// {
//     if(fileinput==null)
//     {
//         return 
//     }
//     let reader = new FileReader()
//     reader.onload = function(e)
//     {
//         let url = '/process/entity_recongnition'
//         var httprequest = new XMLHttpRequest()
//         httprequest.open('POST',url,true)
//         httprequest.setRequestHeader("Content-type","application/x-www-form-urlencoded");
//         httprequest.send(e.target.result)
//         httprequest.onreadystatechange = function(){
//             if(httprequest.status ==200)
//             {
//                 var res = httprequest.responseText
//                 replace_preview("data:image/jepg;base64,"+res)
//             }
//         else
//         {
//         console.log("请求失败了")
//         }
//         } 

//     }
//     reader.readAsDataURL(fileinput)
// }