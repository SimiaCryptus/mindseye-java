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
import com.simiacryptus.ref.lang.ReferenceCounting;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefIntStream;
import com.simiacryptus.ref.wrappers.RefList;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Map;
import java.util.UUID;
import java.util.function.IntFunction;
import java.util.function.IntToDoubleFunction;

@SuppressWarnings("serial")
public class SumReducerLayer extends LayerBase {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(SumReducerLayer.class);

  public SumReducerLayer() {
  }

  protected SumReducerLayer(@Nonnull final JsonObject id) {
    super(id);
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static SumReducerLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new SumReducerLayer(json);
  }

  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    try {
      TensorList temp_62_0003 = inObj[0].getData();
      Result temp_62_0002 = new Result(new TensorArray(RefIntStream.range(0, temp_62_0003.length()).parallel()
          .mapToDouble(RefUtil.wrapInterface((IntToDoubleFunction) dataIndex -> {
            double sum = 0;
            for (@Nonnull final Result element : inObj) {
              TensorList data = element.getData();
              @Nullable
              Tensor tensor = data.get(dataIndex);
              data.freeRef();
              @Nullable final double[] input = tensor.getData();
              tensor.freeRef();
              for (final double element2 : input) {
                sum += element2;
              }
            }
            return sum;
          }, RefUtil.addRefs(inObj))).mapToObj(x -> new Tensor(new double[]{x}, new int[]{1}))
          .toArray(i -> new Tensor[i])), new Result.Accumulator() {
        {
          RefUtil.addRefs(inObj);
        }

        @Override
        public void accept(@Nullable DeltaSet<UUID> buffer, @Nonnull TensorList data) {
          for (@Nonnull final Result in_l : inObj) {
            if (in_l.isAlive()) {
              TensorList data1 = in_l.getData();
              @Nonnull
              TensorArray tensorArray = new TensorArray(RefIntStream.range(0, data1.length()).parallel()
                  .mapToObj(RefUtil.wrapInterface((IntFunction<Tensor>) dataIndex -> {
                    Tensor tensor = data.get(dataIndex);
                    assert 1 == tensor.length() : RefArrays.toString(tensor.getDimensions());
                    @Nonnull final Tensor passback = new Tensor(data1.getDimensions());
                    for (int i = 0; i < Tensor.length(data1.getDimensions()); i++) {
                      passback.set(i, tensor.get(0));
                    }
                    tensor.freeRef();
                    return passback;
                  }, data.addRef(), in_l.addRef(), data1))
                  .toArray(i -> new Tensor[i]));
              in_l.accumulate(buffer == null ? null : buffer.addRef(), tensorArray);
            }
          }
          data.freeRef();
          if (null != buffer)
            buffer.freeRef();
        }

        public @SuppressWarnings("unused")
        void _free() {
          ReferenceCounting.freeRefs(inObj);
        }
      }) {

        {
          RefUtil.addRefs(inObj);
        }

        @Override
        public boolean isAlive() {
          for (@Nonnull final Result element : inObj)
            if (element.isAlive()) {
              return true;
            }
          return false;
        }

        public void _free() {
          ReferenceCounting.freeRefs(inObj);
          super._free();
        }
      };
      temp_62_0003.freeRef();
      return temp_62_0002;
    } finally {
      ReferenceCounting.freeRefs(inObj);
    }
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
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  SumReducerLayer addRef() {
    return (SumReducerLayer) super.addRef();
  }
}
