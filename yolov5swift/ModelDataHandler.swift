//
//  ModelDataHandler.swift
//  yolov5swift
//
//  Created by Dzung Ngo on 25/02/2023.
//

import CoreImage
import TensorFlowLite
import UIKit

typealias fileInfo = (name: String, extension: String)

enum YOLOv5 {
    static let modelInfo: fileInfo = (name: "yolov5", extension: "tflite")
    static let labelsInfo: fileInfo = (name: "label", extension: "txt")
}

class ModelDataHandler: NSObject {
    private var interpreter: Interpreter!
    private var labels: [String] = []
    private let threshold: Float = 0.6
    
    // Model parameters
    let batchSize = 1
    let inputWidth = 640
    let inputHeight = 640
    let inputChannels = 3
    
    private let bgraPixel = (channels: 4, alphaComponent: 3, lastBgrComponent: 2)
    private let rgbPixelChannels = 3
    private let colorStrideValue = 10

    init?(modelFileInfo: fileInfo, labelFileInfo: fileInfo) {
        super.init()
        createInterpreter(modelFileInfo: modelFileInfo)
        loadLabels(labelFileInfo: labelFileInfo)
    }
    
    
    // MARK: - Methods
    
    func runModel(frame pixelBuffer: CVPixelBuffer) -> [Inference]?{
        var inferenceArray: [Inference] = []
        let imageWidth = CVPixelBufferGetWidth(pixelBuffer)
        let imageHeight = CVPixelBufferGetHeight(pixelBuffer)
        
        let sourcePixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer)
        assert(sourcePixelFormat == kCVPixelFormatType_32ARGB ||
               sourcePixelFormat == kCVPixelFormatType_32BGRA ||
               sourcePixelFormat == kCVPixelFormatType_32RGBA)
        
        let imageChannels = 6
        assert(imageChannels >= inputChannels)
        
        // Crops the image to the biggest square in the center and scales it down to model dimensions.
        let scaledSize = CGSize(width: inputWidth, height: inputHeight)
        guard let scaledPixelBuffer = pixelBuffer.resized(to: scaledSize) else {
            return nil
        }
        
        do {
            let inputTensor = try interpreter.input(at: 0)
            
            // Remove the alpha component from the image buffer to get the RGB data.
            guard let rgbData = rgbDataFromBuffer(
                scaledPixelBuffer,
                byteCount: batchSize * inputWidth * inputHeight * inputChannels,
                isModelQuantized: inputTensor.dataType == .uInt8
            ) else {
                print("Failed to convert the image buffer to RGB data.")
                return nil
            }
            
            // Copy the RGB data to the input `Tensor`.
            try interpreter.copy(rgbData, toInputAt: 0)
            
            // Run inference by invoking the `Interpreter`.
            try interpreter.invoke()
            
            let modelOutput: Tensor = try interpreter.output(at: 0)
            
            inferenceArray = formatModelOutput(
                modelOutput: [Float](unsafeData: modelOutput.data) ?? [],
                width: CGFloat(imageWidth),
                height: CGFloat(imageHeight)
            )
    
        } catch let error {
            print("Failed to invoke the interpreter with error: \(error)")
            return nil
        }
   
        return inferenceArray
    }
    
    private func formatModelOutput(modelOutput: [Float], width: CGFloat, height: CGFloat) -> [Inference] {
        var resultsArray: [Inference] = []
        let listOutput = Array(stride(from: 0, to: modelOutput.count, by: 6)).map {
            Array(modelOutput[$0..<min($0 + 6, modelOutput.count)])
        }
        
        for out in listOutput {
            let outputClass: String = "face"
            let score: Float = out[4]
            
            // Translates the detected bounding box to CGRect.
//            var rect: CGRect = CGFloat.zero
//            rect.origin.x = CGFloat(out[0])
//            rect.origin.y = CGFloat(out[1])
//            rect.size.height = CGFloat(out[3]) - rect.origin.y
//            rect.size.width = CGFloat(out[2]) - rect.origin.x
            
            let rect = convertYoloToCGRect(x: out[0], y: out[1], w: out[2], h: out[3], imageWidth: Float(width), imageHeight: Float(height))
            
            // The detected corners are for model dimensions. So we scale the rect with respect to the
            // actual image dimensions.
//            let newRect = rect.applying(CGAffineTransform(scaleX: width, y: height))
            let inference = Inference(confidence: score,
                                      className: outputClass,
                                      rect: rect,
                                      displayColor: UIColor.green)
            
            if inference.confidence > threshold {
                resultsArray.append(inference)
            }
        }
        
        // Sort results in descending order of confidence.
        resultsArray.sort { (first, second) -> Bool in
            return first.confidence  > second.confidence
        }
        
        if resultsArray.count > 0 {
            return [resultsArray[0]]
        } else {
            return []
        }
    }
    
    func convertYoloToCGRect(x: Float, y: Float, w: Float, h: Float, imageWidth: Float, imageHeight: Float) -> CGRect {
        let rectX = (x - w/2) * Float(imageWidth) - 60
        let rectY = (y - h/2) * Float(imageHeight) - 50
        let rectW = w * Float(imageWidth) + 100
        let rectH = h * Float(imageHeight)
        return CGRect(x: CGFloat(rectX), y: CGFloat(rectY), width: CGFloat(rectW), height: CGFloat(rectH))
    }
    
    /** Returns the RGB data representation of the given image buffer with the specified `byteCount`.

     - Parameters
        - buffer: The BGRA pixel buffer to convert to RGB data.
        - byteCount: The expected byte count for the RGB data calculated using the values that the model was trained on: `batchSize * imageWidth * imageHeight * componentsCount`.
        - isModelQuantized: Whether the model is quantized (i.e. fixed point values rather than floating point values).
     
     - Returns: The RGB data representation of the image buffer or `nil` if the buffer could not be converted.
    */
    private func rgbDataFromBuffer(_ buffer: CVPixelBuffer, byteCount: Int, isModelQuantized: Bool) -> Data? {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        guard let mutableRawPointer = CVPixelBufferGetBaseAddress(buffer) else {
            return nil
        }
        assert(CVPixelBufferGetPixelFormatType(buffer) == kCVPixelFormatType_32BGRA)
        let count = CVPixelBufferGetDataSize(buffer)
        let bufferData = Data(bytesNoCopy: mutableRawPointer, count: count, deallocator: .none)
        var rgbBytes = [UInt8](repeating: 0, count: byteCount)
        var pixelIndex = 0
        for component in bufferData.enumerated() {
            let bgraComponent = component.offset % bgraPixel.channels;
            let isAlphaComponent = bgraComponent == bgraPixel.alphaComponent;
            guard !isAlphaComponent else {
                pixelIndex += 1
                continue
            }
            // Swizzle BGR -> RGB.
            let rgbIndex = pixelIndex * rgbPixelChannels + (bgraPixel.lastBgrComponent - bgraComponent)
            rgbBytes[rgbIndex] = component.element
        }
        if isModelQuantized { return Data(_: rgbBytes) }
        return Data(copyingBufferOf: rgbBytes.map { Float($0) / 255.0 })
    }
    
    private func createInterpreter(modelFileInfo: fileInfo) {
        guard let modelPath = Bundle.main.path(
            forResource: modelFileInfo.name,
            ofType: modelFileInfo.extension
        ) else {
            print("Failed to define model path as path: \(modelFileInfo.name).")
            return
        }
        
        // Create Interpreter
        do {
            interpreter = try Interpreter(modelPath: modelPath)
            
            // Allocate memory for the model's input `Tensor`s.
            try interpreter.allocateTensors()
        } catch let error {
            print("Failed to create interpreter: \(error)")
        }
    }
    
    private func loadLabels(labelFileInfo: fileInfo) {
        let filename = labelFileInfo.name
        let fileExtension = labelFileInfo.extension
        
        guard let fileURL = Bundle.main.url(
            forResource: filename,
            withExtension: fileExtension)
        else {
            fatalError("Labels file not found in bundle with name: \(filename).\(fileExtension)")
            
        }
        
        do {
            let contents = try String(contentsOf: fileURL, encoding: .utf8)
            labels = contents.components(separatedBy: .newlines) // ["line1", "", "line2", ...]
        } catch {
            fatalError("Can not read label file with name: \(filename).\(fileExtension)")
        }
    }
}

// MARK: - Extensions

extension Data {
    /// Creates a new buffer by copying the buffer pointer of the given array.
    ///
    /// - Warning: The given array's element type `T` must be trivial in that it can be copied bit
    ///     for bit with no indirection or reference-counting operations; otherwise, reinterpreting
    ///     data from the resulting buffer has undefined behavior.
    /// - Parameter array: An array with elements of type `T`.
    init<T>(copyingBufferOf array: [T]) {
        self = array.withUnsafeBufferPointer(Data.init)
    }
}

extension Array {
    /** Creates a new array from the bytes of the given unsafe data.
     - Warning: The array's `Element` type must be trivial in that it can be copied bit for bit with no indirection or reference-counting operations; otherwise, copying the raw bytes in the `unsafeData`'s buffer to a new array returns an unsafe copy.
     
     - Note: Returns `nil` if `unsafeData.count` is not a multiple of `MemoryLayout<Element>.stride`.
     
     - Parameter unsafeData: The data containing the bytes to turn into an array.
     */
    init?(unsafeData: Data) {
        guard unsafeData.count % MemoryLayout<Element>.stride == 0 else { return nil }
#if swift(>=5.0)
        self = unsafeData.withUnsafeBytes { .init($0.bindMemory(to: Element.self)) }
#else
        self = unsafeData.withUnsafeBytes {
            .init(UnsafeBufferPointer<Element>(
                start: $0,
                count: unsafeData.count / MemoryLayout<Element>.stride
            ))
        }
#endif  // swift(>=5.0)
    }
}
