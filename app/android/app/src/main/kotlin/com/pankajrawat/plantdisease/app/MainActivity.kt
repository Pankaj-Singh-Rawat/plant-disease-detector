package com.pankajrawat.plantdisease.app

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel

class MainActivity : FlutterActivity() {
    private val CHANNEL = "com.pankajrawat.plantdisease/inference"
    private lateinit var classifier: PlantDiseaseClassifier

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)

        classifier = PlantDiseaseClassifier(applicationContext)

        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, CHANNEL).setMethodCallHandler { call, result ->
            if (call.method == "classifyImage") {
                val imagePath = call.argument<String>("imagePath")
                if (imagePath == null) {
                    result.error("INVALID_ARGUMENT", "imagePath is required", null)
                    return@setMethodCallHandler
                }
                try {
                    val bitmap = BitmapFactory.decodeFile(imagePath)
                    val (label, confidence) = classifier.classify(bitmap)
                    result.success(mapOf("label" to label, "confidence" to confidence))
                } catch (e: Exception) {
                    result.error("INFERENCE_ERROR", e.message, null)
                }
            } else {
                result.notImplemented()
            }
        }
    }
}