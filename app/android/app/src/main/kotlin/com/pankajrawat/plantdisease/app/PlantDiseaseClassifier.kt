package com.pankajrawat.plantdisease.app

import android.content.Context
import android.graphics.Bitmap
import org.pytorch.IValue
import org.pytorch.LiteModuleLoader
import org.pytorch.Module
import org.pytorch.Tensor
import org.pytorch.torchvision.TensorImageUtils
import org.json.JSONArray
import java.io.File
import java.io.FileOutputStream

class PlantDiseaseClassifier(context: Context) {

    private val module: Module
    private val labels: List<String>

    // Must match the normalization used in dataset.py exactly
    private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val STD = floatArrayOf(0.229f, 0.224f, 0.225f)

    init {
        module = LiteModuleLoader.load(assetFilePath(context, "plant_disease_model_quantized.ptl"))
        labels = loadLabels(context, "labels.json")
    }

    /** Copies a bundled asset to internal storage and returns its absolute path
     *  (PyTorch's loader needs a real file path, not an asset stream). */
    private fun assetFilePath(context: Context, assetName: String): String {
        val file = File(context.filesDir, assetName)
        if (file.exists() && file.length() > 0) {
            return file.absolutePath
        }
        context.assets.open(assetName).use { input ->
            FileOutputStream(file).use { output ->
                input.copyTo(output)
            }
        }
        return file.absolutePath
    }

    private fun loadLabels(context: Context, assetName: String): List<String> {
        val json = context.assets.open(assetName).bufferedReader().use { it.readText() }
        val array = JSONArray(json)
        return List(array.length()) { i -> array.getString(i) }
    }

    /** Runs inference on a Bitmap, returns Pair(predictedLabel, confidence 0-1). */
    fun classify(bitmap: Bitmap): Pair<String, Float> {
        val resized = Bitmap.createScaledBitmap(bitmap, 224, 224, true)

        val inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
            resized,
            MEAN,
            STD
        )

        val outputTensor: Tensor = module.forward(IValue.from(inputTensor)).toTensor()
        val scores = outputTensor.dataAsFloatArray

        // Softmax to turn raw logits into a confidence percentage
        val maxScore = scores.max()
        val expScores = scores.map { Math.exp((it - maxScore).toDouble()) }
        val sumExp = expScores.sum()
        val probabilities = expScores.map { (it / sumExp).toFloat() }

        var bestIdx = 0
        var bestProb = probabilities[0]
        for (i in probabilities.indices) {
            if (probabilities[i] > bestProb) {
                bestProb = probabilities[i]
                bestIdx = i
            }
        }

        return Pair(labels[bestIdx], bestProb)
    }
}