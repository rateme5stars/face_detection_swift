//
//  Inference.swift
//  yolov5swift
//
//  Created by Dzung Ngo on 25/02/2023.
//

import Foundation
import UIKit

struct Inference {
    let confidence: Float
    let className: String
    let rect: CGRect
    let displayColor: UIColor
}
