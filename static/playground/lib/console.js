(function () {
var conEl = document.createElement('div');
conEl.id='console-log';
document.body.append(conEl);
console.old = console.log;
var old = console.log;
var logger = document.getElementById('console-log');
console.log = function () {
    var content = '';
    for (var i = 0; i < arguments.length; i++) {
    if (typeof arguments[i] == 'object') {
        content += (JSON && JSON.stringify ? JSON.stringify(arguments[i], undefined, 2) : console.old.apply(void 0, arguments)) + '<br />';
    } else {
        content += arguments[i] + '<br />';
    }
    }
    logger.innerHTML += content;
}
console.error = function () {
    var content = '';
    for (var i = 0; i < arguments.length; i++) {
    if (typeof arguments[i] == 'object') {
        content += (JSON && JSON.stringify ? JSON.stringify(arguments[i], undefined, 2) : console.old.apply(void 0, arguments)) + '<br />';
    } else {
        content += arguments[i] + '<br />';
    }
    }
    logger.innerHTML += '<div class="error-div">'+content+'</div>';
}
})();