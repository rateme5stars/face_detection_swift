//
//  Overlay.swift
//  yolov5swift
//
//  Created by Dzung Ngo on 25/02/2023.
//

import UIKit

/**
 This structure holds the display parameters for the overlay to be drawon on a detected object.
 */
struct ObjectOverlay {
  let name: String
  let borderRect: CGRect
  let nameStringSize: CGSize
  let color: UIColor
  let font: UIFont
}
