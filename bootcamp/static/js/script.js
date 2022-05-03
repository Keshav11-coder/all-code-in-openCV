// Socket handler
var gateway = `ws://${window.location.hostname}/ws`;
var websocket;
window.addEventListener('load', onload);


function onload(event) {
    initWebSocket();
}

function getValues(){
    websocket.send("getValues");
}

function initWebSocket() {
    console.log('Trying to open a WebSocket connection…');
    websocket = new WebSocket(gateway);
    websocket.onopen = onOpen;
    websocket.onclose = onClose;
    websocket.onmessage = onMessage;
}

function onOpen(event) {
    console.log('Connection opened');
    getValues();
}

function onClose(event) {
    console.log('Connection closed');
    setTimeout(initWebSocket, 2000);
}

function updateSliderPWM(element) {
    var sliderNumber = element.id.charAt(element.id.length-1);
    var sliderValue = document.getElementById(element.id).value;
    document.getElementById("sliderValue"+sliderNumber).innerHTML = sliderValue;
    console.log(sliderValue);
    websocket.send(sliderNumber+"s"+sliderValue.toString());
}

function onMessage(event) {
    console.log(event.data);
    var myObj = JSON.parse(event.data);
    var keys = Object.keys(myObj);

    for (var i = 0; i < keys.length; i++){
        var key = keys[i];
        document.getElementById(key).innerHTML = myObj[key];
        document.getElementById("slider"+ (i+1).toString()).value = myObj[key];
    }
}


// JoyStick handler
let StickStatus =
{
    xPosition: 0,
    yPosition: 0,
    x: 0,
    y: 0,
    cardinalDirection: "C"
};

/**
 * @desc Principal object that draw a joystick, you only need to initialize the object and suggest the HTML container
 * @costructor
 * @param container {String} - HTML object that contains the Joystick
 * @param parameters (optional) - object with following keys:
 *  title {String} (optional) - The ID of canvas (Default value is 'joystick')
 *  width {Int} (optional) - The width of canvas, if not specified is setted at width of container object (Default value is the width of container object)
 *  height {Int} (optional) - The height of canvas, if not specified is setted at height of container object (Default value is the height of container object)
 *  internalFillColor {String} (optional) - Internal color of Stick (Default value is '#00AA00')
 *  internalLineWidth {Int} (optional) - Border width of Stick (Default value is 2)
 *  internalStrokeColor {String}(optional) - Border color of Stick (Default value is '#003300')
 *  externalLineWidth {Int} (optional) - External reference circonference width (Default value is 2)
 *  externalStrokeColor {String} (optional) - External reference circonference color (Default value is '#008000')
 *  autoReturnToCenter {Bool} (optional) - Sets the behavior of the stick, whether or not, it should return to zero position when released (Default value is True and return to zero)
 * @param callback {StickStatus} - 
 */
var JoyStick = (function(container, parameters, callback)
{
    parameters = parameters || {};
    var title = (typeof parameters.title === "undefined" ? "joystick" : parameters.title),
        width = (typeof parameters.width === "undefined" ? 0 : parameters.width),
        height = (typeof parameters.height === "undefined" ? 0 : parameters.height),
        internalFillColor = (typeof parameters.internalFillColor === "undefined" ? "#00AA00" : parameters.internalFillColor),
        internalLineWidth = (typeof parameters.internalLineWidth === "undefined" ? 2 : parameters.internalLineWidth),
        internalStrokeColor = (typeof parameters.internalStrokeColor === "undefined" ? "#003300" : parameters.internalStrokeColor),
        externalLineWidth = (typeof parameters.externalLineWidth === "undefined" ? 2 : parameters.externalLineWidth),
        externalStrokeColor = (typeof parameters.externalStrokeColor ===  "undefined" ? "#008000" : parameters.externalStrokeColor),
        autoReturnToCenter = (typeof parameters.autoReturnToCenter === "undefined" ? true : parameters.autoReturnToCenter);

    callback = callback || function(StickStatus) {};

    // Create Canvas element and add it in the Container object
    var objContainer = document.getElementById(container);
    
    // Fixing Unable to preventDefault inside passive event listener due to target being treated as passive in Chrome [Thanks to https://github.com/artisticfox8 for this suggestion]
    objContainer.style.touchAction = "none";

    var canvas = document.createElement("canvas");
    canvas.id = title;
    if(width === 0) { width = objContainer.clientWidth; }
    if(height === 0) { height = objContainer.clientHeight; }
    canvas.width = width;
    canvas.height = height;
    objContainer.appendChild(canvas);
    var context=canvas.getContext("2d");

    var pressed = 0; // Bool - 1=Yes - 0=No
    var circumference = 2 * Math.PI;
    var internalRadius = (canvas.width-((canvas.width/2)+10))/2;
    var maxMoveStick = internalRadius + 5;
    var externalRadius = internalRadius + 30;
    var centerX = canvas.width / 2;
    var centerY = canvas.height / 2;
    var directionHorizontalLimitPos = canvas.width / 10;
    var directionHorizontalLimitNeg = directionHorizontalLimitPos * -1;
    var directionVerticalLimitPos = canvas.height / 10;
    var directionVerticalLimitNeg = directionVerticalLimitPos * -1;
    // Used to save current position of stick
    var movedX=centerX;
    var movedY=centerY;

    // Check if the device support the touch or not
    if("ontouchstart" in document.documentElement)
    {
        canvas.addEventListener("touchstart", onTouchStart, false);
        document.addEventListener("touchmove", onTouchMove, false);
        document.addEventListener("touchend", onTouchEnd, false);
    }
    else
    {
        canvas.addEventListener("mousedown", onMouseDown, false);
        document.addEventListener("mousemove", onMouseMove, false);
        document.addEventListener("mouseup", onMouseUp, false);
    }
    // Draw the object
    drawExternal();
    drawInternal();

    /******************************************************
     * Private methods
     *****************************************************/

    /**
     * @desc Draw the external circle used as reference position
     */
    function drawExternal()
    {
        context.beginPath();
        context.arc(centerX, centerY, externalRadius, 0, circumference, false);
        context.lineWidth = externalLineWidth;
        context.strokeStyle = externalStrokeColor;
        context.stroke();
    }

    /**
     * @desc Draw the internal stick in the current position the user have moved it
     */
    function drawInternal()
    {
        context.beginPath();
        if(movedX<internalRadius) { movedX=maxMoveStick; }
        if((movedX+internalRadius) > canvas.width) { movedX = canvas.width-(maxMoveStick); }
        if(movedY<internalRadius) { movedY=maxMoveStick; }
        if((movedY+internalRadius) > canvas.height) { movedY = canvas.height-(maxMoveStick); }
        context.arc(movedX, movedY, internalRadius, 0, circumference, false);
        // create radial gradient
        var grd = context.createRadialGradient(centerX, centerY, 5, centerX, centerY, 200);
        // Light color
        grd.addColorStop(0, internalFillColor);
        // Dark color
        grd.addColorStop(1, internalStrokeColor);
        context.fillStyle = grd;
        context.fill();
        context.lineWidth = internalLineWidth;
        context.strokeStyle = internalStrokeColor;
        context.stroke();
    }

    /**
     * @desc Events for manage touch
     */
    function onTouchStart(event) 
    {
        pressed = 1;
    }

    function onTouchMove(event)
    {
        if(pressed === 1 && event.targetTouches[0].target === canvas)
        {
            movedX = event.targetTouches[0].pageX;
            movedY = event.targetTouches[0].pageY;
            // Manage offset
            if(canvas.offsetParent.tagName.toUpperCase() === "BODY")
            {
                movedX -= canvas.offsetLeft;
                movedY -= canvas.offsetTop;
            }
            else
            {
                movedX -= canvas.offsetParent.offsetLeft;
                movedY -= canvas.offsetParent.offsetTop;
            }
            // Delete canvas
            context.clearRect(0, 0, canvas.width, canvas.height);
            // Redraw object
            drawExternal();
            drawInternal();

            // Set attribute of callback
            StickStatus.xPosition = movedX;
            StickStatus.yPosition = movedY;
            StickStatus.x = (100*((movedX - centerX)/maxMoveStick)).toFixed();
            StickStatus.y = ((100*((movedY - centerY)/maxMoveStick))*-1).toFixed();
            StickStatus.cardinalDirection = getCardinalDirection();
            callback(StickStatus);
        }
    } 

    function onTouchEnd(event) 
    {
        pressed = 0;
        // If required reset position store variable
        if(autoReturnToCenter)
        {
            movedX = centerX;
            movedY = centerY;
        }
        // Delete canvas
        context.clearRect(0, 0, canvas.width, canvas.height);
        // Redraw object
        drawExternal();
        drawInternal();

        // Set attribute of callback
        StickStatus.xPosition = movedX;
        StickStatus.yPosition = movedY;
        StickStatus.x = (100*((movedX - centerX)/maxMoveStick)).toFixed();
        StickStatus.y = ((100*((movedY - centerY)/maxMoveStick))*-1).toFixed();
        StickStatus.cardinalDirection = getCardinalDirection();
        callback(StickStatus);
    }

    /**
     * @desc Events for manage mouse
     */
    function onMouseDown(event) 
    {
        pressed = 1;
    }

    /* To simplify this code there was a new experimental feature here: https://developer.mozilla.org/en-US/docs/Web/API/MouseEvent/offsetX , but it present only in Mouse case not metod presents in Touch case :-( */
    function onMouseMove(event) 
    {
        if(pressed === 1)
        {
            movedX = event.pageX;
            movedY = event.pageY;
            // Manage offset
            if(canvas.offsetParent.tagName.toUpperCase() === "BODY")
            {
                movedX -= canvas.offsetLeft;
                movedY -= canvas.offsetTop;
            }
            else
            {
                movedX -= canvas.offsetParent.offsetLeft;
                movedY -= canvas.offsetParent.offsetTop;
            }
            // Delete canvas
            context.clearRect(0, 0, canvas.width, canvas.height);
            // Redraw object
            drawExternal();
            drawInternal();

            // Set attribute of callback
            StickStatus.xPosition = movedX;
            StickStatus.yPosition = movedY;
            StickStatus.x = (100*((movedX - centerX)/maxMoveStick)).toFixed();
            StickStatus.y = ((100*((movedY - centerY)/maxMoveStick))*-1).toFixed();
            StickStatus.cardinalDirection = getCardinalDirection();
            callback(StickStatus);
        }
    }

    function onMouseUp(event) 
    {
        pressed = 0;
        // If required reset position store variable
        if(autoReturnToCenter)
        {
            movedX = centerX;
            movedY = centerY;
        }
        // Delete canvas
        context.clearRect(0, 0, canvas.width, canvas.height);
        // Redraw object
        drawExternal();
        drawInternal();

        // Set attribute of callback
        StickStatus.xPosition = movedX;
        StickStatus.yPosition = movedY;
        StickStatus.x = (100*((movedX - centerX)/maxMoveStick)).toFixed();
        StickStatus.y = ((100*((movedY - centerY)/maxMoveStick))*-1).toFixed();
        StickStatus.cardinalDirection = getCardinalDirection();
        callback(StickStatus);
    }

    function getCardinalDirection()
    {
        let result = "";
        let orizontal = movedX - centerX;
        let vertical = movedY - centerY;
        
        if(vertical >= directionVerticalLimitNeg && vertical <= directionVerticalLimitPos)
        {
            result = "C";
        }
        if(vertical < directionVerticalLimitNeg)
        {
            result = "N";
        }
        if(vertical > directionVerticalLimitPos)
        {
            result = "S";
        }
        
        if(orizontal < directionHorizontalLimitNeg)
        {
            if(result === "C")
            { 
                result = "W";
            }
            else
            {
                result += "W";
            }
        }
        if(orizontal > directionHorizontalLimitPos)
        {
            if(result === "C")
            { 
                result = "E";
            }
            else
            {
                result += "E";
            }
        }
        
        return result;
    }

    /******************************************************
     * Public methods
     *****************************************************/

    /**
     * @desc The width of canvas
     * @return Number of pixel width 
     */
    this.GetWidth = function () 
    {
        return canvas.width;
    };

    /**
     * @desc The height of canvas
     * @return Number of pixel height
     */
    this.GetHeight = function () 
    {
        return canvas.height;
    };

    /**
     * @desc The X position of the cursor relative to the canvas that contains it and to its dimensions
     * @return Number that indicate relative position
     */
    this.GetPosX = function ()
    {
        return movedX;
    };

    /**
     * @desc The Y position of the cursor relative to the canvas that contains it and to its dimensions
     * @return Number that indicate relative position
     */
    this.GetPosY = function ()
    {
        return movedY;
    };

    /**
     * @desc Normalizzed value of X move of stick
     * @return Integer from -100 to +100
     */
    this.GetX = function ()
    {
        return (100*((movedX - centerX)/maxMoveStick)).toFixed();
    };

    /**
     * @desc Normalizzed value of Y move of stick
     * @return Integer from -100 to +100
     */
    this.GetY = function ()
    {
        return ((100*((movedY - centerY)/maxMoveStick))*-1).toFixed();
    };

    /**
     * @desc Get the direction of the cursor as a string that indicates the cardinal points where this is oriented
     * @return String of cardinal point N, NE, E, SE, S, SW, W, NW and C when it is placed in the center
     */
    this.GetDir = function()
    {
        return getCardinalDirection();
    };
});


// ### Streetview Buttons ###
L.StreetView = L.Control.extend({
  options: {
    google: true,
    bing: true,
    yandex: true,
    mapillary: true,
    mapillaryId: null,
    openstreetcam: true,
    mosatlas: true
  },

  providers: [
    ['google', 'GSV', 'Google Street View', false,
      'https://www.google.com/maps?layer=c&cbll={lat},{lon}'],
    ['bing', 'Bing', 'Bing StreetSide',
      L.latLngBounds([[25, -168], [71.4, 8.8]]),
      'https://www.bing.com/maps?cp={lat}~{lon}&lvl=19&style=x&v=2'],
    ['yandex', 'ЯП', 'Yandex Panoramas',
      L.latLngBounds([[35.6, 18.5], [72, 180]]),
      'https://yandex.ru/maps/?panorama%5Bpoint%5D={lon},{lat}'],
    ['mapillary', 'Mplr', 'Mapillary Photos', false,
      'https://a.mapillary.com/v3/images?client_id={id}&closeto={lon},{lat}&lookat={lon},{lat}'],
    ['openstreetcam', 'OSC', 'OpenStreetCam', false,
      'lat={lat}&lng={lon}&distance=50'],
    ['mosatlas', 'Мос', 'Панорамы из Атласа Москвы',
      L.latLngBounds([[55.113, 36.708], [56.041, 38]]),
      'http://atlas.mos.ru/?lang=ru&z=9&ll={lon}%2C{lat}&pp={lon}%2C{lat}'],
  ],

  onAdd: function(map) {
    this._container = L.DomUtil.create('div', 'leaflet-bar');
    this._buttons = [];

    for (var i = 0; i < this.providers.length; i++)
      this._addProvider(this.providers[i]);

    map.on('moveend', function() {
      if (!this._fixed)
        this._update(map.getCenter());
    }, this);
    this._update(map.getCenter());
    return this._container;
  },

  fixCoord: function(latlon) {
    this._update(latlon);
    this._fixed = true;
  },

  releaseCoord: function() {
    this._fixed = false;
    this._update(this._map.getCenter());
  },

  _addProvider: function(provider) {
    if (!this.options[provider[0]])
      return;
    if (provider[0] == 'mapillary' && !this.options.mapillaryId)
      return;
    var button = L.DomUtil.create('a');
    button.innerHTML = provider[1];
    button.title = provider[2];
    button._bounds = provider[3];
    button._template = provider[4];
    button.href = '#';
    button.target = 'streetview';
    button.style.padding = '0 8px';
    button.style.width = 'auto';

    // Some buttons require complex logic
    if (provider[0] == 'mapillary') {
      button._needUrl = false;
      L.DomEvent.on(button, 'click', function(e) {
        if (button._href) {
          this._ajaxRequest(
            button._href.replace(/{id}/, this.options.mapillaryId),
            function(data) {
              if (data && data.features && data.features[0].properties) {
                var photoKey = data.features[0].properties.key,
                    url = 'https://www.mapillary.com/map/im/{key}'.replace(/{key}/, photoKey);
                window.open(url, button.target);
              }
            }
          );
        }
        return L.DomEvent.preventDefault(e);
      }, this);
    } else if (provider[0] == 'openstreetcam') {
      button._needUrl = false;
      L.DomEvent.on(button, 'click', function(e) {
        if (button._href) {
          this._ajaxRequest(
            'http://openstreetcam.org/nearby-tracks',
            function(data) {
              if (data && data.osv && data.osv.sequences) {
                var seq = data.osv.sequences[0],
                    url = 'https://www.openstreetcam.org/details/'+seq.sequence_id+'/'+seq.sequence_index;
                window.open(url, button.target);
              }
            },
            button._href
          );
        }
        return L.DomEvent.preventDefault(e);
      }, this);
    } else
      button._needUrl = true;

    // Overriding some of the leaflet styles
    button.style.display = 'inline-block';
    button.style.border = 'none';
    button.style.borderRadius = '0 0 0 0';
    this._buttons.push(button);
  },

  _update: function(center) {
    if (!center)
      return;
    var last;
    for (var i = 0; i < this._buttons.length; i++) {
      var b = this._buttons[i],
          show = !b._bounds || b._bounds.contains(center),
          vis = this._container.contains(b);

      if (show && !vis) {
        ref = last ? last.nextSibling : this._container.firstChild;
        this._container.insertBefore(b, ref);
      } else if (!show && vis) {
        this._container.removeChild(b);
        return;
      }
      last = b;

      var tmpl = b._template;
      tmpl = tmpl
        .replace(/{lon}/g, L.Util.formatNum(center.lng, 6))
        .replace(/{lat}/g, L.Util.formatNum(center.lat, 6));
      if (b._needUrl)
        b.href = tmpl;
      else
        b._href = tmpl;
    }
  },

  _ajaxRequest: function(url, callback, post_data) {
    if (window.XMLHttpRequest === undefined)
      return;
    var req = new XMLHttpRequest();
    req.open(post_data ? 'POST' : "GET", url);
    if (post_data)
      req.setRequestHeader('Content-Type', 'application/x-www-form-urlencoded');
    req.onreadystatechange = function() {
      if (req.readyState === 4 && req.status == 200) {
        var data = (JSON.parse(req.responseText));
        callback(data);
      }
    };
    req.send(post_data);
  }
});

L.streetView = function(options) {
  return new L.StreetView(options);
}
