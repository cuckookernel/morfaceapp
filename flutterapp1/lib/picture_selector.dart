import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'package:image_picker/image_picker.dart';
import 'dart:io';
import 'main.dart';

class PictureSelectorCard2 extends StatefulWidget {
  final String imgLabel;
  PictureSelectorCard2(this.imgLabel) : super();
  @override
  _PictureSelectorState createState() => _PictureSelectorState(this.imgLabel);
}

class _PictureSelectorState extends State<PictureSelectorCard2> {
  final String imgLabel;
  Future<PickedFile> imageFile;
  ImagePicker imgPicker = ImagePicker();

  _PictureSelectorState(this.imgLabel);

  pickImageFromGallery(ImageSource source) {
    setState(() {
      imageFile = imgPicker.getImage(source: source);
    });
  }

  @override
  Widget build(BuildContext context) =>
      Row( children:  <Widget>[
              Expanded(
                  child: showImage()
              ),
              //showImage(),
              Column(
                  children: [
                    IconButton(
                        icon: Icon( Icons.camera, color: Colors.blue ),
                        onPressed: () {
                          pickImageFromGallery(ImageSource.camera);
                        }
                    ),
                    IconButton(
                        icon: Icon( Icons.add_a_photo, color: Colors.blue ),
                        onPressed: () {
                          pickImageFromGallery(ImageSource.gallery);
                        })
                  ]
              )
            ]
      );

  Widget showImage() {
    return FutureBuilder<PickedFile>(
      future: imageFile,
      builder: (BuildContext context, AsyncSnapshot<PickedFile> snapshot) {
        if (snapshot.connectionState == ConnectionState.done &&
            snapshot.data != null) {

          var imageFile = File(snapshot.data.path);
          var model = Provider.of<AppStateModel>(context);

          if( imgLabel == "A" ) {
            model.setImageFileA(imageFile);
          } else if( imgLabel == "B") {
            model.setImageFileB(imageFile);
          }

          return Image.file( imageFile, width: 200, height: 200);
        } else if (snapshot.error != null) {
          return const Text(
            'Error Picking Image',
            textAlign: TextAlign.center,
          );
        } else {
          return const Text(
            'No Image Selected',
            textAlign: TextAlign.center,
          );
        }
      },
    );
  }
}


/*
class PictureSelectorCard extends StatelessWidget {
  @override
  Widget build(BuildContext context) =>
      Row( children:
        <Widget>[
          Expanded(
              child: Image.asset( 'images/no_pic.png')
          ),
          Column(
              children: [
                IconButton(
                  icon: Icon( Icons.camera, color: Colors.blue ) ),
                IconButton(
                  icon: Icon( Icons.add_a_photo, color: Colors.blue ) )
              ]
          )
        ]
      );
}
*/
