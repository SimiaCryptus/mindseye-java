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
import com.simiacryptus.ref.wrappers.RefString;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map;
import java.util.UUID;
import java.util.function.IntFunction;

@SuppressWarnings("serial")
public class LinearActivationLayer extends LayerBase {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(LinearActivationLayer.class);
  @Nullable
  private final Tensor weights;

  public LinearActivationLayer() {
    super();
    Tensor temp_04_0001 = new Tensor(2);
    weights = temp_04_0001.addRef();
    temp_04_0001.freeRef();
    weights.set(0, 1.);

    weights.set(1, 0.);
  }

  protected LinearActivationLayer(@Nonnull final JsonObject json, Map<CharSequence, byte[]> resources) {
    super(json);
    Tensor temp_04_0002 = Tensor.fromJson(json.get("weights"), resources);
    weights = temp_04_0002 == null ? null : temp_04_0002.addRef();
    if (null != temp_04_0002)
      temp_04_0002.freeRef();
  }

  public double getBias() {
    assert weights != null;
    return weights.get(1);
  }

  public void setBias(double bias) {
    if (!Double.isFinite(bias))
      throw new IllegalArgumentException();
    assert weights != null;
    weights.set(1, bias);
  }

  @Nullable
  @Override
  public String getName() {
    String eqStr = isFrozen() ? "== " : "=> ";
    assert weights != null;
    if (weights.get(0) == 1.0) {
      return eqStr + RefString.format("x + %.1e", weights.get(1)) + (isFrozen() ? "" : "!");
    } else if (weights.get(1) == 0.0) {
      return eqStr + RefString.format("%.1e x", weights.get(0)) + (isFrozen() ? "" : "!");
    } else {
      return eqStr + RefString.format("%.1e x + %.1e", weights.get(0), weights.get(1));
    }
  }

  public double getScale() {
    assert weights != null;
    return weights.get(0);
  }

  public void setScale(double scale) {
    if (!Double.isFinite(scale))
      throw new IllegalArgumentException();
    assert weights != null;
    weights.set(0, scale);
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static LinearActivationLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new LinearActivationLayer(json, rs);
  }

  @Nonnull
  @Override
  public Result eval(@Nullable final Result... inObj) {
    assert inObj != null;
    final Result in0 = inObj[0].addRef();
    RefUtil.freeRefs(inObj);
    final TensorList inData = in0.getData();
    final int itemCnt = inData.length();
    assert weights != null;
    final double scale = weights.get(0);
    final double bias = weights.get(1);
    final LinearActivationLayer linearActivationLayer = LinearActivationLayer.this.addRef();
    try {
      Result.Accumulator accumulator = new Result.Accumulator() {
        {
          inData.addRef();
          weights.addRef();
          in0.addRef();
          linearActivationLayer.addRef();
        }

        @Override
        public void accept(@Nonnull DeltaSet<UUID> buffer, @Nonnull TensorList delta) {
          if (!LinearActivationLayer.this.isFrozen()) {
            RefIntStream.range(0, delta.length()).forEach(RefUtil.wrapInterface(dataIndex -> {
                  @Nullable
                  Tensor deltaT = delta.get(dataIndex);
                  @Nullable
                  Tensor inputT = inData.get(dataIndex);
                  @Nullable final double[] deltaData = deltaT.getData();
                  deltaT.freeRef();
                  @Nullable final double[] inputData = inputT.getData();
                  inputT.freeRef();
                  @Nonnull final Tensor weightDelta = new Tensor(weights.getDimensions());
                  for (int i = 0; i < deltaData.length; i++) {
                    weightDelta.add(0, deltaData[i] * inputData[inputData.length == 1 ? 0 : i]);
                    weightDelta.add(1, deltaData[i]);
                  }
                  Delta<UUID> temp_04_0006 = buffer.get(linearActivationLayer.getId(), weights.getData());
                  assert temp_04_0006 != null;
                  temp_04_0006.addInPlace(weightDelta.getData());
                  temp_04_0006.freeRef();
                  weightDelta.freeRef();
                }, buffer.addRef(), inData.addRef(),
                linearActivationLayer.addRef(),
                delta.addRef()));
          }
          if (in0.isAlive()) {
            @Nonnull final TensorList tensorList = new TensorArray(RefIntStream.range(0, delta.length())
                .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) dataIndex -> {
                  @Nullable
                  Tensor tensor = delta.get(dataIndex);
                  @Nullable final double[] deltaData = tensor.getData();
                  tensor.freeRef();
                  @Nonnull final Tensor passback = new Tensor(inData.getDimensions());
                  for (int i = 0; i < passback.length(); i++) {
                    passback.set(i, deltaData[i] * weights.getData()[0]);
                  }
                  return passback;
                }, inData.addRef(), delta.addRef()))
                .toArray(i -> new Tensor[i]));
            in0.accumulate(buffer.addRef(), tensorList);
          }
          delta.freeRef();
          buffer.freeRef();
        }

        public @SuppressWarnings("unused")
        void _free() {
          super._free();
          inData.freeRef();
          weights.freeRef();
          in0.freeRef();
          linearActivationLayer.freeRef();
        }
      };
      TensorArray data = new TensorArray(RefIntStream.range(0, itemCnt)
          .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) dataIndex -> {
            Tensor tensor = inData.get(dataIndex);
            Tensor tensor1 = tensor.map(v -> {
              final double r = scale * v + bias;
              return Double.isFinite(r) ? r : 0;
            });
            tensor.freeRef();
            return tensor1;
          }, inData.addRef())).toArray(i -> new Tensor[i]));
      return new Result(data, accumulator) {
        {
          in0.addRef();
        }
        @Override
        public boolean isAlive() {
          return in0.isAlive() || !isFrozen();
        }

        @Override
        public void _free() {
          in0.freeRef();
          super._free();
        }
      };
    } finally {
      linearActivationLayer.freeRef();
      inData.freeRef();
      in0.freeRef();
    }
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, @Nonnull DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    assert weights != null;
    json.add("weights", weights.getJson(resources, dataSerializer));
    return json;
  }

  @Nonnull
  @Override
  public RefList<double[]> state() {
    assert weights != null;
    return RefArrays.asList(weights.getData());
  }

  public void _free() {
    if (null != weights)
      weights.freeRef();
    super._free();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  LinearActivationLayer addRef() {
    return (LinearActivationLayer) super.addRef();
  }

}
