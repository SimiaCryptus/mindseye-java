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
import java.util.UUID;
import java.util.function.IntFunction;

@SuppressWarnings("serial")
public abstract class SimpleActivationLayer<T extends SimpleActivationLayer<T>> extends LayerBase {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(SigmoidActivationLayer.class);

  public SimpleActivationLayer() {
    super();
    this.frozen = true;
  }

  protected SimpleActivationLayer(@Nonnull final JsonObject id) {
    super(id);
  }

  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    final Result inObj0 = inObj[0].addRef();
    RefUtil.freeRef(inObj);
    final TensorList indata0 = inObj0.getData();
    final int itemCnt = indata0.length();
    assert 0 < itemCnt;
    @Nonnull final Tensor inputGradientA[] = new Tensor[itemCnt];
    try {
      TensorArray data = new TensorArray(RefIntStream.range(0, itemCnt).parallel()
          .mapToObj(RefUtil.wrapInterface((IntFunction<Tensor>) dataIndex -> {
            @Nullable final Tensor input = indata0.get(dataIndex);
            @Nonnull final Tensor output = new Tensor(indata0.getDimensions());
            int length = input.length();
            @Nonnull final Tensor inputGradient = new Tensor(length);
            @Nonnull final double[] results = new double[2];
            double[] inputData = input.getData();
            for (int i = 0; i < length; i++) {
              eval(inputData[i], results);
              inputGradient.set(i, results[1]);
              output.set(i, results[0]);
            }
            RefUtil.set(inputGradientA, dataIndex, inputGradient);
            input.freeRef();
            return output;
          }, indata0))
          .toArray(Tensor[]::new));
      final boolean inObj0Alive = inObj0.isAlive();
      Result.Accumulator accumulator = new Result.Accumulator() {
        {
          inObj0.addRef();
          RefUtil.addRefs(inputGradientA);
        }

        @Override
        public void accept(@Nullable DeltaSet<UUID> buffer, @Nonnull TensorList data) {
          data.assertAlive();
          if (null != buffer) buffer.assertAlive();
          if (inObj0Alive) {
            inObj0.accumulate(buffer == null ? null : buffer.addRef(),
                new TensorArray(RefIntStream.range(0, itemCnt).parallel()
                    .mapToObj(RefUtil.wrapInterface((IntFunction<Tensor>) dataIndex -> {
                      @Nonnull final Tensor passback = new Tensor(data.getDimensions());
                      Tensor tensor1 = null == inputGradientA ? null : inputGradientA[dataIndex].addRef();
                      @Nullable final double[] gradientData = null == tensor1 ? null : tensor1.getData();
                      @Nullable
                      Tensor tensor = data.get(dataIndex);
                      RefIntStream.range(0, passback.length()).forEach(RefUtil.wrapInterface(i -> {
                        final double v = null == gradientData ? 0 : gradientData[i];
                        if (Double.isFinite(v)) {
                          passback.set(i, tensor.get(i) * v);
                        }
                      }, passback.addRef(), tensor));
                      if(null != tensor1) tensor1.freeRef();
                      return passback;
                    }, data, RefUtil.addRefs(inputGradientA)))
                    .toArray(Tensor[]::new)));
          } else {
            data.freeRef();
          }
          if (null != buffer)
            buffer.freeRef();
        }

        public @SuppressWarnings("unused")
        void _free() {
          super._free();
          RefUtil.freeRef(inputGradientA);
          inObj0.freeRef();
        }
      };
      return new Result(data, accumulator) {
        @Override
        public boolean isAlive() {
          return inObj0Alive;
        }

        @Override
        public void _free() {
          super._free();
        }
      };
    } finally {
      RefUtil.freeRef(inputGradientA);
      inObj0.freeRef();
    }
  }

  @Nonnull
  @Override
  public RefList<double[]> state() {
    return RefArrays.asList();
  }

  public @SuppressWarnings("unused")
  void _free() { super._free(); }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  SimpleActivationLayer<T> addRef() {
    return (SimpleActivationLayer<T>) super.addRef();
  }

  protected abstract void eval(final double x, double[] results);

}
