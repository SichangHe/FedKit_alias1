//
//  MLFlwrClient.swift
//
//
//  Created by Daniel Nugraha on 18.01.23.
//  Simplified by Steven Hé to adapt to fed_kit.
//

import CoreML
import Foundation
import NIOCore
import NIOPosix
import os

public enum MLTask {
    case train
    case test
}

enum MLClientErr: Error {
    case NoParamUpdate
    case ParamsNil
    case ParamNotMultiArray
    case UpdateParamShapeMismatch(Int, Int)
}

public class MLClient {
    let layers: [Layer]
    var parameters: [MLMultiArray]?
    var dataLoader: MLDataLoader
    var compiledModelUrl: URL
    var tempModelUrl: URL
    private var paramUpdate: [[Float]]?

    let log = logger(String(describing: MLClient.self))

    init(_ layers: [Layer], _ dataLoader: MLDataLoader, _ compiledModelUrl: URL) {
        self.layers = layers
        self.dataLoader = dataLoader
        self.compiledModelUrl = compiledModelUrl

        let modelFileName = compiledModelUrl.deletingPathExtension().lastPathComponent
        tempModelUrl = appDirectory.appendingPathComponent("temp\(modelFileName).mlmodelc")
    }

    func getParameters() async throws -> [[Float]] {
        if parameters == nil {
            try await fit()
        }
        guard let parameters else {
            throw MLClientErr.ParamsNil
        }
        return try parameters.map { layer in
            let pointer = try UnsafeBufferPointer<Float>(layer)
            return Array(pointer)
        }
    }

    func updateParameters(parameters: [[Float]]) {
        paramUpdate = parameters
    }

    func fit() async throws {
        let config = try config()
        let updateContext = try await updateModelAsync(
            forModelAt: compiledModelUrl, trainingData: dataLoader.trainBatchProvider, configuration: config
        )
        parameters = try layers.map { layer in
            let paramKey = MLParameterKey.weights.scoped(to: layer.name)
            guard let weightsMultiArray = try updateContext.model.parameterValue(for: paramKey) as? MLMultiArray else {
                throw MLClientErr.ParamNotMultiArray
            }
            return weightsMultiArray
        }
        logMLMultiArrayShapes(array: parameters!)
        try saveModel(updateContext)
    }

    func evaluate() async throws -> (Double, Double) {
        let config = try config()
        config.parameters![MLParameterKey.epochs] = 1
        let updateContext = try await updateModelAsync(
            forModelAt: compiledModelUrl, trainingData: dataLoader.testBatchProvider, configuration: config
        )
        let loss = updateContext.metrics[.lossValue] as! Double
        return (loss, (1.0 - loss) * 100)
    }

    /// Guarantee that the config returned has non-nil `parameters`.
    private func config() throws -> MLModelConfiguration {
        let config = MLModelConfiguration()
        if config.parameters == nil {
            config.parameters = [:]
        }
        var mlMultiArrays = [MLMultiArray]()
        if let paramUpdate {
            for (index, weightsArray) in paramUpdate.enumerated() {
                let layer = layers[index]
                let mlMultiArray = try makeMlMultiArray(layer, weightsArray)
                mlMultiArrays.append(mlMultiArray)
                let paramKey = MLParameterKey.weights.scoped(to: layer.name)
                config.parameters![paramKey] = mlMultiArray
            }
            self.paramUpdate = nil
        }
        logMLMultiArrayShapes(array: mlMultiArrays)
        return config
    }

    private func validateLayerShape(_ layer: Layer, _ weightsArray: [Float]) throws {
        let expectedShape = layer.shape.reduce(1, *)
        if expectedShape != weightsArray.count {
            throw MLClientErr.UpdateParamShapeMismatch(expectedShape, weightsArray.count)
        }
    }

    private func makeMlMultiArray(_ layer: Layer, _ weightsArray: [Float]) throws -> MLMultiArray {
        try validateLayerShape(layer, weightsArray)
        var array = try MLMultiArray(shape: layer.shape as [NSNumber], dataType: .float)
        for (index, param) in weightsArray.enumerated() {
            array[index] = param as NSNumber
        }
        return array
    }

    private func logMLMultiArrayShapes(array: [MLMultiArray]) {
        let parameterShape = array.map { $0.shape }
        log.error("MLClient: MLMultiArray shapes: \(parameterShape)")
    }

    private func saveModel(_ updateContext: MLUpdateContext) throws {
        let updatedModel = updateContext.model
        let fileManager = FileManager.default
        try fileManager.createDirectory(at: tempModelUrl, withIntermediateDirectories: true, attributes: nil)
        try updatedModel.write(to: tempModelUrl)
        _ = try fileManager.replaceItemAt(compiledModelUrl, withItemAt: tempModelUrl)
    }
}
