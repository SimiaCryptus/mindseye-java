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
import com.simiacryptus.ref.lang.RefAware;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefIntStream;
import com.simiacryptus.ref.wrappers.RefList;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.UUID;

@SuppressWarnings("serial")
public abstract @RefAware
class SimpleActivationLayer<T extends SimpleActivationLayer<T>>
    extends LayerBase {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(SigmoidActivationLayer.class);

  public SimpleActivationLayer() {
    super();
    this.frozen = true;
  }

  protected SimpleActivationLayer(@Nonnull final JsonObject id) {
    super(id);
  }

  public static @SuppressWarnings("unused")
  SimpleActivationLayer[] addRefs(SimpleActivationLayer[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(SimpleActivationLayer::addRef)
        .toArray((x) -> new SimpleActivationLayer[x]);
  }

  public static @SuppressWarnings("unused")
  SimpleActivationLayer[][] addRefs(SimpleActivationLayer[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(SimpleActivationLayer::addRefs)
        .toArray((x) -> new SimpleActivationLayer[x][]);
  }

  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    final TensorList indata0 = inObj[0].getData();
    final int itemCnt = indata0.length();
    assert 0 < itemCnt;
    @Nonnull final Tensor inputGradientA[] = new Tensor[itemCnt];
    return new Result(
        new TensorArray(RefIntStream.range(0, itemCnt).parallel().mapToObj(dataIndex -> {
          @Nullable final Tensor input = indata0.get(dataIndex);
          @Nonnull final Tensor output = new Tensor(indata0.getDimensions());
          @Nonnull final Tensor inputGradient = new Tensor(input.length());
          inputGradientA[dataIndex] = inputGradient;
          @Nonnull final double[] results = new double[2];
          for (int i = 0; i < input.length(); i++) {
            eval(input.getData()[i], results);
            inputGradient.set(i, results[1]);
            output.set(i, results[0]);
          }
          return output;
        }).toArray(i -> new Tensor[i])), new Result.Accumulator() {
      @Override
      public void accept(DeltaSet<UUID> buffer, TensorList data) {
        if (inObj[0].isAlive()) {
          @Nonnull
          TensorArray tensorArray = new TensorArray(
              RefIntStream.range(0, itemCnt).parallel().mapToObj(dataIndex -> {
                @Nonnull final Tensor passback = new Tensor(data.getDimensions());
                @Nullable final double[] gradientData = inputGradientA[dataIndex].getData();
                @Nullable
                Tensor tensor = data.get(dataIndex);
                RefIntStream.range(0, passback.length()).forEach(i -> {
                  final double v = gradientData[i];
                  if (Double.isFinite(v)) {
                    passback.set(i, tensor.get(i) * v);
                  }
                });
                return passback;
              }).toArray(i -> new Tensor[i]));
          inObj[0].accumulate(buffer, tensorArray);
        }
      }
    }) {

      @Override
      public boolean isAlive() {
        return inObj[0].isAlive();
      }

      public void _free() {
      }
    };
  }

  @Nonnull
  @Override
  public RefList<double[]> state() {
    return RefArrays.asList();
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  public @Override
  @SuppressWarnings("unused")
  SimpleActivationLayer<T> addRef() {
    return (SimpleActivationLayer<T>) super.addRef();
  }

  protected abstract void eval(final double x, double[] results);

}
