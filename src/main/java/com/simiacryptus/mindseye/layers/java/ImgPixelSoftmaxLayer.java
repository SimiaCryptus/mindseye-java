/*
 * Copyright (c) 2019 by Andrew Charneski.
 *
 * The author licenses this file to you under the
 * Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance
 * with the License.  You may obtain a copy
 * of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.simiacryptus.mindseye.layers.java;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefIntStream;
import com.simiacryptus.ref.wrappers.RefList;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map;
import java.util.UUID;
import java.util.function.IntFunction;

@SuppressWarnings("serial")
public class ImgPixelSoftmaxLayer extends LayerBase {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(ImgPixelSoftmaxLayer.class);

  public ImgPixelSoftmaxLayer() {
    super();
  }

  protected ImgPixelSoftmaxLayer(@Nonnull final JsonObject json) {
    super(json);
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static ImgPixelSoftmaxLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new ImgPixelSoftmaxLayer(json);
  }

  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    assert 1 == inObj.length;
    Result result = eval(inObj[0].addRef());
    RefUtil.freeRef(inObj);
    return result;
  }

  @Nonnull
  public Result eval(@Nonnull final Result input) {
    final TensorList inputData = input.getData();
    int[] inputDims = inputData.getDimensions();
    assert 3 == inputDims.length;
    final int inputBands = inputDims[2];
    final int width = inputDims[0];
    final int height = inputDims[1];
    TensorArray maxima = new TensorArray(inputData.stream().map(inputTensor -> {
      Tensor tensor = new Tensor(width, height, 1);
      tensor.setByCoord(RefUtil.wrapInterface(c -> {
        return RefIntStream.range(0, inputBands).mapToDouble(RefUtil.wrapInterface(band -> {
          int[] coords = c.getCoords();
          return inputTensor.get(coords[0], coords[1], band);
        }, inputTensor == null ? null : inputTensor.addRef())).max().getAsDouble();
      }, inputTensor));
      return tensor;
    }).toArray(Tensor[]::new));
    TensorArray exps = new TensorArray(RefIntStream.range(0, inputData.length())
        .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) index -> {
          final Tensor inputTensor = inputData.get(index);
          Tensor maxTensor = maxima.get(index);
          Tensor tensor = new Tensor(inputDims);
          tensor.setByCoord(RefUtil.wrapInterface(c -> {
            int[] coords = c.getCoords();
            return Math.exp(inputTensor.get(c) - maxTensor.get(coords[0], coords[1], 0));
          }, maxTensor, inputTensor));
          return tensor;
        }, inputData.addRef(), maxima.addRef()))
        .toArray(Tensor[]::new));
    maxima.freeRef();
    TensorArray sums = new TensorArray(exps.stream().map(expTensor -> {
      Tensor tensor = new Tensor(width, height, 1);
      tensor.setByCoord(RefUtil.wrapInterface(c -> {
        return RefIntStream.range(0, inputBands).mapToDouble(RefUtil.wrapInterface(band -> {
          int[] coords = c.getCoords();
          return expTensor.get(coords[0], coords[1], band);
        }, expTensor == null ? null : expTensor.addRef())).sum();
      }, expTensor));
      return tensor;
    }).toArray(Tensor[]::new));
    TensorArray output = new TensorArray(RefIntStream.range(0, inputData.length())
        .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) index -> {
          Tensor sumTensor = sums.get(index);
          Tensor expTensor = exps.get(index);
          Tensor tensor = new Tensor(inputDims);
          tensor.setByCoord(RefUtil.wrapInterface(c -> {
            int[] coords = c.getCoords();
            return expTensor.get(c) / sumTensor.get(coords[0], coords[1], 0);
          }, sumTensor, expTensor));
          return tensor;
        }, sums.addRef(), exps.addRef())).toArray(Tensor[]::new));
    boolean alive = input.isAlive();
    Accumulator accumulator = new Accumulator(sums, inputData, exps, width, height, inputBands, inputDims, input.getAccumulator(), input.isAlive());
    input.freeRef();
    return new Result(output, accumulator, alive);
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    return super.getJsonStub();
  }

  @Nonnull
  @Override
  public RefList<double[]> state() {
    return RefArrays.asList();
  }

  public @SuppressWarnings("unused")
  void _free() {
    super._free();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  ImgPixelSoftmaxLayer addRef() {
    return (ImgPixelSoftmaxLayer) super.addRef();
  }

  private static class Accumulator extends Result.Accumulator {

    private final TensorArray sums;
    private final TensorList inputData;
    private final TensorArray exps;
    private final int width;
    private final int height;
    private final int inputBands;
    private final int[] inputDims;
    private Result.Accumulator accumulator;
    private boolean alive;

    public Accumulator(TensorArray sums, TensorList inputData, TensorArray exps, int width, int height, int inputBands, int[] inputDims, Result.Accumulator accumulator, boolean alive) {
      this.sums = sums;
      this.inputData = inputData;
      this.exps = exps;
      this.width = width;
      this.height = height;
      this.inputBands = inputBands;
      this.inputDims = inputDims;
      this.accumulator = accumulator;
      this.alive = alive;
    }

    @Override
    public void accept(@Nullable DeltaSet<UUID> buffer, @Nonnull TensorList delta) {
      if (alive) {
        TensorArray dots = new TensorArray(RefIntStream.range(0, inputData.length())
            .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) index -> {
              final Tensor deltaTensor = delta.get(index);
              Tensor expTensor = exps.get(index);
              Tensor tensor = new Tensor(width, height, 1);
              tensor.setByCoord(RefUtil.wrapInterface(c -> {
                return RefIntStream.range(0, inputBands).mapToDouble(RefUtil.wrapInterface(band -> {
                      int[] coords = c.getCoords();
                      return expTensor.get(coords[0], coords[1], band)
                          * deltaTensor.get(coords[0], coords[1], band);
                    }, deltaTensor.addRef(),
                    expTensor.addRef())).sum();
              }, deltaTensor, expTensor));
              return tensor;
            }, delta.addRef(), exps.addRef()))
            .toArray(Tensor[]::new));

        TensorArray passback = new TensorArray(RefIntStream.range(0, inputData.length())
            .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) index -> {
                  final Tensor deltaTensor = delta.get(index);
                  final Tensor expTensor = exps.get(index);
                  Tensor sumTensor = sums.get(index);
                  Tensor dotTensor = dots.get(index);
                  Tensor tensor = new Tensor(inputDims);
                  tensor.setByCoord(RefUtil.wrapInterface(c -> {
                    int[] coords = c.getCoords();
                    double sum = sumTensor.get(coords[0], coords[1], 0);
                    double dot = dotTensor.get(coords[0], coords[1], 0);
                    double deltaValue = deltaTensor.get(c);
                    double expValue = expTensor.get(c);
                    return (sum * deltaValue - dot) * expValue / (sum * sum);
                  }, deltaTensor, expTensor, dotTensor, sumTensor));
                  return tensor;
                }, sums.addRef(), dots,
                delta.addRef(), exps.addRef()))
            .toArray(Tensor[]::new));
        DeltaSet<UUID> buffer1 = buffer == null ? null : buffer.addRef();
        this.accumulator.accept(buffer1, passback);
      }
      delta.freeRef();
      if (null != buffer)
        buffer.freeRef();
    }

    public @SuppressWarnings("unused")
    void _free() {
      super._free();
      accumulator.freeRef();
      sums.freeRef();
      inputData.freeRef();
      exps.freeRef();
    }
  }
}
