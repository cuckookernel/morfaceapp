import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'dart:developer';
import 'package:image_picker/image_picker.dart';


import 'dart:io' as io;
import 'dart:convert' as cv;
import 'package:http/http.dart' as http;

// import './app_state.dart';
import './picture_selector.dart';


// const SERVER_HOST_PORT = "http://127.0.0.1:5000";
const SERVER_HOST_PORT = "http://192.168.1.5:8000";

void main() {
  runApp(
      ChangeNotifierProvider(
          create: (context) => AppStateModel(),
          child: MyApp()
      )
  );
}

class MyApp extends StatelessWidget {
  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Morface App',
      theme: ThemeData(
        // This is the theme of your application.
        //
        // Try running your application with "flutter run". You'll see the
        // application has a blue toolbar. Then, without quitting the app, try
        // changing the primarySwatch below to Colors.green and then invoke
        // "hot reload" (press "r" in the console where you ran "flutter run",
        // or simply save your changes to "hot reload" in a Flutter IDE).
        // Notice that the counter didn't reset back to zero; the application
        // is not restarted.
        primarySwatch: Colors.blue,
        // This makes the visual density adapt to the platform that you run
        // the app on. For desktop platforms, the controls will be smaller and
        // closer together (more dense) than on mobile platforms.
        visualDensity: VisualDensity.adaptivePlatformDensity,
      ),
      home: MyHomePage(),
    );
  }
}


class MyHomePage extends StatelessWidget {
  MyHomePage({Key key}) : super(key: key);

  // This widget is the home page of your application. It is stateful, meaning
  // that it has a State object (defined below) that contains fields that affect
  // how it looks.

  // This class is the configuration for the state. It holds the values (in this
  // case the title) provided by the parent (in this case the App widget) and
  // used by the build method of the State. Fields in a Widget subclass are
  // always marked "final".
  @override
  Widget build(BuildContext context) {
    // This method is rerun every time setState is called, for instance as done
    // by the _incrementCounter method above.
    //
    // The Flutter framework has been optimized to make rerunning build methods
    // fast, so that you can just rebuild anything that needs updating rather
    // than having to individually change instances of widgets.
    return Consumer<AppStateModel>( builder:
        (context, appStateModel, child) {
      return Scaffold(
          appBar: AppBar(
            // Here we take the value from the MyHomePage object that was created by
            // the App.build method, and use it to set our appbar title.
            title: Text("Home"),
          ),
          body: Center(
              child: // Expanded( child:
              Column(
                //
                // Invoke "debug painting" (press "p" in the console, choose the
                // "Toggle Debug Paint" action from the Flutter Inspector in Android
                // Studio, or the "Toggle Debug Paint" command in Visual Studio Code)
                // to see the wireframe for each widget.
                //
                // Column has various properties to control how it sizes itself and
                // how it positions its children. Here we use mainAxisAlignment to
                // center the children vertically; the main axis here is the vertical
                // axis because Columns are vertical (the cross axis would be
                // horizontal).
                mainAxisAlignment: MainAxisAlignment.center,
                mainAxisSize: MainAxisSize.max,
                children: <Widget>[
                  PictureSelectorCard("A"),
                  PictureSelectorCard("B")
                ],
              )
            //),
          ),
          //),
          bottomNavigationBar:  Container(
              child: Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    IconButton(
                        icon: Icon(Icons.people_alt),
                        iconSize: 50,
                        onPressed: null
                    ),
                    IconButton(
                        icon: Icon(Icons.local_movies),
                        iconSize: 50,
                        // onPressed: appStateModel.getFaceCombinationResult
                        onPressed: () {
                          appStateModel.getFaceCombinationResult();

                          Navigator.push(context,
                              MaterialPageRoute(
                                  builder: (context) => CombImagePage()
                              )
                          );
                        })
                  ]
              ))
      );
    }
    );
  }
}

class AppStateModel extends ChangeNotifier {
  Map<String, ImageState> imageStates = {"A": ImageState(), "B": ImageState()};
  Image combImage;
  double _combinationProgress = 0.0;
  get combinationProgress => _combinationProgress;

  void setFutureImage(String imageLabel, Future<PickedFile> pickedImageFile) {
    log("setFutureImageFile called $imageLabel");

    imageStates[imageLabel].reset();
    imageStates[imageLabel].futureImageFile = pickedImageFile;
    notifyListeners();
  }

  void setImageFile(String imageLabel, io.File imageFile) {
    log("setImageFile called $imageLabel $imageFile");
    imageStates[imageLabel].file = imageFile;
    notifyListeners();
  }

  void getFaceCombinationResult() async {
    _combinationProgress = 0;
    // notifyListeners();

    for( String label in ['A', 'B'] ) {
      if( imageStates[label].key == null ) {
        imageStates[label].key = await uploadImageAsync(imageStates[label].file);
        log("$label.key: ${imageStates[label].key}");
      }

      _combinationProgress = 0.1;
      notifyListeners();

      if (imageStates[label].faceIdx == null) {
        log("Before calling detectFaces $label: ${imageStates[label].key}");

        await detectFacesAsync(imageStates[label].key);
        imageStates[label].faceIdx = 0;
      }
      _combinationProgress = 0.2;

      notifyListeners();
      // TODO: check that numFaces for each of the two images is not 0.
      // If numFaces == 0, should tell user no faces detected on image and stop
      // If numFaces > 1  need to ask user which face to use

      imageStates[label].landmarks = await imageStates[label].detectLandmarksAsync();


    }

    combImage = await combineFacesAsync(0.5);
    _combinationProgress = 0.9;
    notifyListeners();

  }


  Future<int> detectFacesAsync(String imgKey) async {
    log("calling detectFacesEndpoint $imgKey");

    final response = await http.get( SERVER_HOST_PORT + '/detect_faces?img_key=$imgKey' );

    if (response.statusCode == 200) {
      final jsonObj =cv.jsonDecode(response.body);
      final len = jsonObj['face_bboxes'].length;

      log("face bboxes returned: $len");
      return len;

    } else {
      throw Exception('call to /detect_faces');
    }
  }

  Future<Image> combineFacesAsync(double lambda ) async {
    log("calling /combine_faces ${imageStates['A'].key} ${imageStates['B'].key}");

    final dataObj = {
      "img1_key": imageStates['A'].key,
      "landmarks1": imageStates['A'].landmarks.points,
      "face_bbox1": imageStates['A'].landmarks.bbox,
      "img2_key": imageStates['B'].key,
      "landmarks2": imageStates['B'].landmarks.points,
      "face_bbox2": imageStates['B'].landmarks.bbox,
      "lambd":  lambda,
    };

    final jsonText = cv.jsonEncode( dataObj );

    log("combineFacesAsync: request json_text has length ${jsonText.length}");

    final response1 = await http.post(
        SERVER_HOST_PORT + '/combine_faces',
        headers: { "Content-Type": 'application/json' },
        body: jsonText
    );

    if (response1.statusCode == 200) {
      final jsonObj  = cv.jsonDecode(response1.body) as Map<String, dynamic>;
      final resultKey = jsonObj['img_key'];

      final url = SERVER_HOST_PORT + '/download_image?img_key=$resultKey';
      // final response2 = await http.get( SERVER_HOST_PORT + '/download_image?img_key=$resultKey');
      // final bytes = response2.bodyBytes;
      // log("combineFacesAsync: response2: $response2 result_key=$resultKey  ${bytes.length} bytes");
      return Image.network(url);
    } else {
      throw Exception('call to /combine_faces failed');
    }
  }


}

class ImageState {
  Future<PickedFile> futureImageFile;
  io.File file;
  String key;
  int faceIdx;
  Landmarks landmarks;

  void reset() {
    file = null;
    key = null;
    faceIdx = null;
    landmarks = null;
  }

  Future<Landmarks> detectLandmarksAsync() async {
      return _detectLandmarksAsync(key, faceIdx);
  }
}

Future<String> uploadImageAsync(io.File imgFile) async {
  log("uploading image $imgFile");

  final data = await imgFile.readAsBytes();

  final ext = imgFile.path.split("\.").last;

  final response = await http.post(
      SERVER_HOST_PORT + '/upload_image',
      headers: {
        "Content-Length": data.length.toString(),
        "Content-Type": 'image/' + ext},
      body: data
  );

  if (response.statusCode == 200) {
    // If the server did return a 200 OK response,
    // then parse the JSON.
    final imgKey = cv.jsonDecode(response.body)['img_key'];
    log('image key returned by server: $imgKey');
    return imgKey;
  } else {
    // If the server did not return a 200 OK response,
    // then throw an exception.
    throw Exception('Failed to upload image');
  }
}


Future<Landmarks> _detectLandmarksAsync(String imgKey, int faceIndex) async {
  log("calling /detect_landmarks $imgKey $faceIndex");

  final response = await http.get( SERVER_HOST_PORT + '/detect_landmarks'
      '?img_key=$imgKey&face_idx=$faceIndex' );

  if (response.statusCode == 200) {
    final jsonObj = cv.jsonDecode(response.body) as Map<String, dynamic>;

    log("landmarks returned: ${(jsonObj['landmarks'] as List).length}");
    return Landmarks(jsonObj['landmarks'] as List,
        jsonObj['new_bbox'] as Map<String, dynamic>);

  } else {
    throw Exception('call to /detect_landmarks failed');
  }
}

class Landmarks {
  List<dynamic> points;
  Map<String, dynamic> bbox;

  Landmarks(this.points, this.bbox);
}



class CombImagePage extends StatelessWidget {
  CombImagePage() : super();

  @override
  Widget build(BuildContext context) {
    return Scaffold(
        appBar: AppBar(
          title: Text("Combined Faces"),
        ),
        body: Center(
            child:
            Consumer<AppStateModel>( builder:
              (context, appStateModel, _ ) => Center(
                child: SizedBox(
                width: 100,
                height: 100,
                child: LinearProgressIndicator(
                  minHeight: 20,
                  value: appStateModel.combinationProgress,
                ),
              ),
              )
            )
        )
    );
  }
}