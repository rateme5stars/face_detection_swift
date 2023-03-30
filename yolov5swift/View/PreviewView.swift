//
//  StreamView.swift
//  yolov5swift
//
//  Created by Dzung Ngo on 25/02/2023.
//

import UIKit
import AVFoundation

/**
 The camera frame is displayed on this view.
 */
class StreamView: UIView {

  var streamLayer: AVCaptureVideoPreviewLayer {
    guard let layer = layer as? AVCaptureVideoPreviewLayer else {
      fatalError("Layer expected is of type VideoPreviewLayer")
    }
    return layer
  }

  var session: AVCaptureSession? {
    get {
      return previewLayer.session
    }
    set {
      previewLayer.session = newValue
    }
  }

  override class var layerClass: AnyClass {
    return AVCaptureVideoPreviewLayer.self
  }
}
