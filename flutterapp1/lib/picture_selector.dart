import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'package:image_picker/image_picker.dart';
import 'dart:io';
import 'main.dart';

class PictureSelectorCard extends StatelessWidget {
  final String imgLabel;
  final ImagePicker imgPicker = ImagePicker();

  PictureSelectorCard(this.imgLabel) : super();

  pickImageFromGallery(AppStateModel appStateModel, ImageSource source) {
    appStateModel.setFutureImage(imgLabel, imgPicker.getImage(source: source));
  }

  @override
  Widget build(BuildContext context) {
    var appStateModel = Provider.of<AppStateModel>(context);

    return Row(
        children:  <Widget>[
          Expanded(
              child: showImage(appStateModel)
          ),
          //showImage(),
          Column(
              children: [
                IconButton(
                    icon: Icon( Icons.camera, color: Colors.blue ),
                    onPressed: () {
                      pickImageFromGallery(appStateModel, ImageSource.camera);
                    }
                ),
                IconButton(
                    icon: Icon( Icons.add_a_photo, color: Colors.blue ),
                    onPressed: () {
                      pickImageFromGallery(appStateModel, ImageSource.gallery);
                    })
              ]
          )
        ]
    );
  }

  Widget showImage(AppStateModel appStateModel) {

    return FutureBuilder<PickedFile>(
      future: appStateModel.imageStates['A'].futureImageFile,
      builder: (BuildContext context, AsyncSnapshot<PickedFile> snapshot) {
        if (snapshot.connectionState == ConnectionState.done &&
            snapshot.data != null) {

          var imageFile = File(snapshot.data.path);
          appStateModel.setImageFile(imgLabel, imageFile);

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
