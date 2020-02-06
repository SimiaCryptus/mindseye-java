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
import com.simiacryptus.ref.lang.RecycleBin;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefIntStream;
import com.simiacryptus.ref.wrappers.RefList;
import com.simiacryptus.ref.wrappers.RefString;
import com.simiacryptus.util.JsonUtil;
import com.simiacryptus.util.Util;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map;
import java.util.UUID;
import java.util.function.DoubleSupplier;
import java.util.function.Function;
import java.util.function.IntToDoubleFunction;

@SuppressWarnings("serial")
public class ImgBandScaleLayer extends LayerBase {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(ImgBandScaleLayer.class);
  @Nullable
  private final double[] weights;

  protected ImgBandScaleLayer() {
    super();
    weights = null;
  }

  public ImgBandScaleLayer(@Nullable final double... bands) {
    super();
    weights = bands;
  }

  protected ImgBandScaleLayer(@Nonnull final JsonObject json) {
    super(json);
    weights = JsonUtil.getDoubleArray(json.getAsJsonArray("bias"));
  }

  @Nullable
  public double[] getWeights() {
    assert weights != null;
    if (!RefArrays.stream(weights).allMatch(Double::isFinite)) {
      throw new IllegalStateException(RefArrays.toString(weights));
    }
    return weights;
  }

  public void setWeights(@Nonnull IntToDoubleFunction f) {
    @Nullable final double[] bias = getWeights();
    assert bias != null;
    for (int i = 0; i < bias.length; i++) {
      bias[i] = f.applyAsDouble(i);
    }
    assert RefArrays.stream(bias).allMatch(Double::isFinite);
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static ImgBandScaleLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new ImgBandScaleLayer(json);
  }

  public void addWeights(@Nonnull DoubleSupplier f) {
    Util.add(f, getWeights());
  }

  @Nonnull
  @Override
  public Result eval(@Nullable final Result... inObj) {
    assert inObj != null;
    Result temp_50_0008 = eval(inObj[0].addRef());
    RefUtil.freeRef(inObj);
    return temp_50_0008;
  }

  @Nonnull
  public Result eval(@Nonnull final Result input) {
    @Nullable final double[] weights = getWeights();
    final TensorList inData = input.getData();
    @Nullable
    Function<Tensor, Tensor> tensorTensorFunction = tensor -> {
      if (tensor.getDimensions().length != 3) {
        IllegalArgumentException temp_50_0003 = new IllegalArgumentException(
            RefArrays.toString(tensor.getDimensions()));
        tensor.freeRef();
        throw temp_50_0003;
      }
      assert weights != null;
      if (tensor.getDimensions()[2] != weights.length) {
        IllegalArgumentException temp_50_0004 = new IllegalArgumentException(RefString.format(
            "%s: %s does not have %s bands", getName(), RefArrays.toString(tensor.getDimensions()), weights.length));
        tensor.freeRef();
        throw temp_50_0004;
      }
      Tensor temp_50_0002 = tensor.mapCoords(RefUtil.wrapInterface(c -> tensor.get(c) * weights[c.getCoords()[2]],
          tensor.addRef()));
      tensor.freeRef();
      return temp_50_0002;
    };
    Tensor[] data = inData.stream().parallel().map(tensorTensorFunction).toArray(Tensor[]::new);
    final ImgBandScaleLayer imgBandScaleLayer = ImgBandScaleLayer.this.addRef();
    try {
      return new Result(new TensorArray(RefUtil.addRefs(data)), new Result.Accumulator() {
        {
          input.addRef();
          imgBandScaleLayer.addRef();
          inData.addRef();
        }

        @Override
        public void accept(@Nonnull DeltaSet<UUID> buffer, @Nonnull TensorList delta) {
          if (!ImgBandScaleLayer.this.isFrozen()) {
            final Delta<UUID> deltaBuffer = buffer.get(imgBandScaleLayer.getId(), weights);
            RefIntStream.range(0, delta.length()).forEach(RefUtil.wrapInterface(index -> {
                  @Nonnull
                  int[] dimensions = delta.getDimensions();
                  int z = dimensions[2];
                  int y = dimensions[1];
                  int x = dimensions[0];
                  final double[] array = RecycleBin.DOUBLES.obtain(z);
                  Tensor deltaTensor = delta.get(index);
                  @Nullable final double[] deltaArray = deltaTensor.getData();
                  deltaTensor.freeRef();
                  Tensor inputTensor = inData.get(index);
                  @Nullable final double[] inputData = inputTensor.getData();
                  inputTensor.freeRef();
                  for (int i = 0; i < z; i++) {
                    for (int j = 0; j < y * x; j++) {
                      //array[i] += deltaArray[i + z * j];
                      array[i] += deltaArray[i * x * y + j] * inputData[i * x * y + j];
                    }
                  }
                  assert RefArrays.stream(array).allMatch(Double::isFinite);
                  assert deltaBuffer != null;
                  deltaBuffer.addInPlace(array);
                  RecycleBin.DOUBLES.recycle(array, array.length);
                }, inData.addRef(), delta.addRef(),
                deltaBuffer == null ? null : deltaBuffer.addRef()));
            if (null != deltaBuffer)
              deltaBuffer.freeRef();
          }
          if (input.isAlive()) {
            Tensor[] tensors = delta.stream().map(t -> {
              Tensor temp_50_0007 = t.mapCoords(RefUtil.wrapInterface(c -> {
                    assert weights != null;
                    return t.get(c) * weights[c.getCoords()[2]];
                  },
                  t.addRef()));
              t.freeRef();
              return temp_50_0007;
            }).toArray(Tensor[]::new);
            @Nonnull
            TensorArray tensorArray = new TensorArray(RefUtil.addRefs(tensors));
            RefUtil.freeRef(tensors);
            input.accumulate(buffer.addRef(), tensorArray);
          }
          delta.freeRef();
          buffer.freeRef();
        }

        public @SuppressWarnings("unused")
        void _free() {
          super._free();
          input.freeRef();
          imgBandScaleLayer.freeRef();
          inData.freeRef();
        }
      }) {

        {
          input.addRef();
        }

        @Override
        public boolean isAlive() {
          return input.isAlive() || !isFrozen();
        }

        public void _free() {
          input.freeRef();
          super._free();
        }
      };
    } finally {
      input.freeRef();
      imgBandScaleLayer.freeRef();
      RefUtil.freeRef(data);
      inData.freeRef();
    }
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    json.add("bias", JsonUtil.getJson(getWeights()));
    return json;
  }

  public void set(@Nonnull double[] ds) {
    @Nullable final double[] bias = getWeights();
    for (int i = 0; i < ds.length; i++) {
      assert bias != null;
      bias[i] = ds[i];
    }
    assert bias != null;
    assert RefArrays.stream(bias).allMatch(Double::isFinite);
  }

  @Nonnull
  @Override
  public RefList<double[]> state() {
    return RefArrays.asList(getWeights());
  }

  public @SuppressWarnings("unused")
  void _free() {
    super._free();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  ImgBandScaleLayer addRef() {
    return (ImgBandScaleLayer) super.addRef();
  }
}
