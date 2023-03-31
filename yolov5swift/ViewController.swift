//
//  ViewController.swift
//  yolov5swift
//
//  Created by Dzung Ngo on 25/02/2023.
//

import UIKit
import TensorFlowLite

class ViewController: UIViewController {
    var previewView: PreviewView!
    var overlayView: OverlayView!
    private lazy var cameraManager = CameraManager(previewView: previewView)
    private var modelDataHandler: ModelDataHandler? = ModelDataHandler(
        modelFileInfo: YOLOv5.modelInfo,
        labelFileInfo: YOLOv5.labelsInfo
    )
    
    private let displayFont = UIFont.systemFont(ofSize: 14.0, weight: .medium)
    private let edgeOffset: CGFloat = 2.0
    private let labelOffset: CGFloat = 10.0
    private let delayBetweenInferencesMs: Double = 50
    private var previousInferenceTimeMs: TimeInterval = Date.distantPast.timeIntervalSince1970 * 100

    override func viewDidLoad() {
        super.viewDidLoad()
        configUI()
        applyConstraints()
        guard modelDataHandler != nil else { fatalError("Failed to load model") }
        
        cameraManager.delegate = self
        overlayView.clearsContextBeforeDrawing = true
    }
    
    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }
    
    override func viewWillAppear(_ animated: Bool) {
      super.viewWillAppear(animated)
      cameraManager.checkCameraConfigurationAndStartSession()
    }

    override func viewWillDisappear(_ animated: Bool) {
      super.viewWillDisappear(animated)
      cameraManager.stopSession()
    }
    
    override var preferredStatusBarStyle: UIStatusBarStyle {
        return .lightContent
    }
    
    // MARK: - Methods
    
    private func configUI() {
        view.backgroundColor = .red
        previewView = PreviewView(frame: view.bounds)

        previewView.previewLayer.connection?.videoOrientation = .portrait
        previewView.previewLayer.connection?.automaticallyAdjustsVideoMirroring = false
//        previewView.previewLayer.connection?.isVideoMirrored = false
        previewView.previewLayer.transform = CATransform3DMakeScale(-1, 1, 1)
        
        overlayView = OverlayView(frame: view.bounds)
        overlayView.backgroundColor = UIColor.clear
        view.addSubview(previewView)
        view.addSubview(overlayView)
    }
    
    private func applyConstraints() {
        previewView.translatesAutoresizingMaskIntoConstraints = false
        overlayView.translatesAutoresizingMaskIntoConstraints = false
        
        let previewViewConstraints = [
            previewView.topAnchor.constraint(equalTo: view.topAnchor),
            previewView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            previewView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            previewView.bottomAnchor.constraint(equalTo: view.bottomAnchor)
        ]
        
        let overlayViewConstraints = [
            overlayView.topAnchor.constraint(equalTo: view.topAnchor),
            overlayView.bottomAnchor.constraint(equalTo: view.bottomAnchor),
            overlayView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            overlayView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
        ]
        NSLayoutConstraint.activate(previewViewConstraints)
        NSLayoutConstraint.activate(overlayViewConstraints)
    }
}

// MARK: CameraFeedManagerDelegate Methods
extension ViewController: CameraManagerDelegate {
    
    func didOutput(pixelBuffer: CVPixelBuffer) {
        runModel(onPixelBuffer: pixelBuffer)
    }
    
    // MARK: Session Handling Alerts
    func sessionRunTimeErrorOccured() {
        //
        //    // Handles session run time error by updating the UI and providing a button if session can be manually resumed.
        //    self.resumeButton.isHidden = false
    }
    
    func sessionWasInterrupted(canResumeManually resumeManually: Bool) {
        //
        //    // Updates the UI when session is interupted.
        //    if resumeManually {
        //      self.resumeButton.isHidden = false
        //    }
        //    else {
        //      self.cameraUnavailableLabel.isHidden = false
        //    }
    }
    
    func sessionInterruptionEnded() {
        //
        //    // Updates UI once session interruption has ended.
        //    if !self.cameraUnavailableLabel.isHidden {
        //      self.cameraUnavailableLabel.isHidden = true
        //    }
        //
        //    if !self.resumeButton.isHidden {
        //      self.resumeButton.isHidden = true
        //    }
    }
    
    func presentVideoConfigurationErrorAlert() {
        
        //    let alertController = UIAlertController(title: "Confirguration Failed", message: "Configuration of camera has failed.", preferredStyle: .alert)
        //    let okAction = UIAlertAction(title: "OK", style: .cancel, handler: nil)
        //    alertController.addAction(okAction)
        //
        //    present(alertController, animated: true, completion: nil)
    }
    
    func presentCameraPermissionsDeniedAlert() {
        
        //    let alertController = UIAlertController(title: "Camera Permissions Denied", message: "Camera permissions have been denied for this app. You can change this by going to Settings", preferredStyle: .alert)
        //
        //    let cancelAction = UIAlertAction(title: "Cancel", style: .cancel, handler: nil)
        //    let settingsAction = UIAlertAction(title: "Settings", style: .default) { (action) in
        //
        //      UIApplication.shared.open(URL(string: UIApplication.openSettingsURLString)!, options: [:], completionHandler: nil)
        //    }
        
        //    alertController.addAction(cancelAction)
        //    alertController.addAction(settingsAction)
        //
        //    present(alertController, animated: true, completion: nil)
        
    }
    
    /**
     This method runs the live camera pixelBuffer through tensorFlow to get the result.
     */
    func runModel(onPixelBuffer pixelBuffer: CVPixelBuffer) {
        // Run the live camera pixelBuffer through tensorFlow to get the result
//        let currentTimeMs = Date().timeIntervalSince1970 * 100
//        guard (currentTimeMs - previousInferenceTimeMs) >= delayBetweenInferencesMs else { return }
//        previousInferenceTimeMs = currentTimeMs
        let first = Date().timeIntervalSince1970
        guard let inferences = self.modelDataHandler?.runModel(frame: pixelBuffer) else { return }
        print(Date().timeIntervalSince1970 - first)
        
        
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        
        DispatchQueue.main.async {
            // Draws the bounding boxes and displays class names and confidence scores.
            self.drawAfterPerformingCalculations(onInferences: inferences, withImageSize: CGSize(width: CGFloat(width), height: CGFloat(height)))
        }
    }
    
    /**
     This method takes the results, translates the bounding box rects to the current view, draws the bounding boxes, classNames and confidence scores of inferences.
     */
    func drawAfterPerformingCalculations(onInferences inferences: [Inference], withImageSize imageSize:CGSize) {
        
        self.overlayView.objectOverlays = []
        self.overlayView.setNeedsDisplay()
        
        guard !inferences.isEmpty else {
            return
        }
        
        var objectOverlays: [ObjectOverlay] = []
        for inference in inferences {
            
            // Translates bounding box rect to current view.
            var convertedRect = inference.rect.applying(CGAffineTransform(scaleX: self.overlayView.bounds.size.width / imageSize.width, y: self.overlayView.bounds.size.height / imageSize.height))
            
            if convertedRect.origin.x < 0 {
                convertedRect.origin.x = self.edgeOffset
            }
            
            if convertedRect.origin.y < 0 {
                convertedRect.origin.y = self.edgeOffset
            }
            
            if convertedRect.maxY > self.overlayView.bounds.maxY {
                convertedRect.size.height = self.overlayView.bounds.maxY - convertedRect.origin.y - self.edgeOffset
            }
            
            if convertedRect.maxX > self.overlayView.bounds.maxX {
                convertedRect.size.width = self.overlayView.bounds.maxX - convertedRect.origin.x - self.edgeOffset
            }
            
            let confidenceValue = Int(inference.confidence * 100.0)
            let string = "\(inference.className)  (\(confidenceValue)%)"
            
            let size = string.size(usingFont: self.displayFont)
            
            let objectOverlay = ObjectOverlay(name: string, borderRect: convertedRect, nameStringSize: size, color: inference.displayColor, font: self.displayFont)
            
            objectOverlays.append(objectOverlay)
        }
        
        // Hands off drawing to the OverlayView
        self.draw(objectOverlays: objectOverlays)
        
    }
    
    /** Calls methods to update overlay view with detected bounding boxes and class names.
     */
    func draw(objectOverlays: [ObjectOverlay]) {
        
        self.overlayView.objectOverlays = objectOverlays
        self.overlayView.setNeedsDisplay()
    }
}

