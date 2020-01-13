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

@SuppressWarnings("serial")
public class ProductLayer extends LayerBase {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(ProductLayer.class);

  public ProductLayer() {
  }

  protected ProductLayer(@Nonnull final JsonObject id) {
    super(id);
  }

  @SuppressWarnings("unused")
  public static ProductLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new ProductLayer(json);
  }

  public static @SuppressWarnings("unused") ProductLayer[] addRefs(ProductLayer[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ProductLayer::addRef).toArray((x) -> new ProductLayer[x]);
  }

  public static @SuppressWarnings("unused") ProductLayer[][] addRefs(ProductLayer[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ProductLayer::addRefs)
        .toArray((x) -> new ProductLayer[x][]);
  }

  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    final Result in0 = inObj[0].addRef();
    assert RefArrays.stream(Result.addRefs(inObj)).mapToInt(x -> {
      TensorList temp_49_0005 = x.getData();
      int temp_49_0001 = temp_49_0005.length();
      if (null != temp_49_0005)
        temp_49_0005.freeRef();
      if (null != x)
        x.freeRef();
      return temp_49_0001;
    }).distinct().count() == 1 : RefArrays.toString(RefArrays.stream(Result.addRefs(inObj)).mapToInt(x -> {
      TensorList temp_49_0006 = x.getData();
      int temp_49_0002 = temp_49_0006.length();
      if (null != temp_49_0006)
        temp_49_0006.freeRef();
      if (null != x)
        x.freeRef();
      return temp_49_0002;
    }).toArray());
    TensorList temp_49_0007 = in0.getData();
    @Nonnull
    final double[] sum_A = new double[temp_49_0007.length()];
    if (null != temp_49_0007)
      temp_49_0007.freeRef();
    TensorList temp_49_0008 = in0.getData();
    final Tensor[] outputA = RefIntStream.range(0, temp_49_0008.length())
        .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) dataIndex -> {
          double sum = 1;
          for (@Nonnull
          final Result input : inObj) {
            TensorList temp_49_0009 = input.getData();
            Tensor tensor = temp_49_0009.get(dataIndex);
            if (null != temp_49_0009)
              temp_49_0009.freeRef();
            @Nullable
            final double[] tensorData = tensor.getData();
            if (null != tensor)
              tensor.freeRef();
            for (final double element2 : tensorData) {
              sum *= element2;
            }
          }
          sum_A[dataIndex] = sum;
          return new Tensor(new double[] { sum }, 1);
        }, Result.addRefs(inObj))).toArray(i -> new Tensor[i]);
    if (null != temp_49_0008)
      temp_49_0008.freeRef();
    if (null != in0)
      in0.freeRef();
    try {
      try {
        return new Result(new TensorArray(Tensor.addRefs(outputA)), new Result.Accumulator() {
          {
            Result.addRefs(inObj);
          }

          @Override
          public void accept(DeltaSet<UUID> buffer, TensorList delta) {
            for (@Nonnull
            final Result input : inObj) {
              if (input.isAlive()) {
                TensorList data = input.getData();
                input.accumulate(buffer == null ? null : buffer.addRef(),
                    new TensorArray(RefIntStream.range(0, delta.length())
                        .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) dataIndex -> {
                          Tensor dataTensor = delta.get(dataIndex);
                          Tensor lTensor = data.get(dataIndex);
                          @Nonnull
                          final Tensor passback = new Tensor(lTensor.getDimensions());
                          for (int i = 0; i < lTensor.length(); i++) {
                            double d = lTensor.getData()[i];
                            double deltaV = dataTensor.get(0);
                            RefUtil.freeRef(passback.set(i, d == 0 ? 0 : (deltaV * sum_A[dataIndex] / d)));
                          }
                          if (null != lTensor)
                            lTensor.freeRef();
                          if (null != dataTensor)
                            dataTensor.freeRef();
                          return passback;
                        }, data == null ? null : data.addRef(), delta == null ? null : delta.addRef()))
                        .toArray(i -> new Tensor[i])));
                if (null != data)
                  data.freeRef();
              }
            }
            if (null != delta)
              delta.freeRef();
            if (null != buffer)
              buffer.freeRef();
          }

          public @SuppressWarnings("unused") void _free() {
            ReferenceCounting.freeRefs(inObj);
          }
        }) {

          {
            Result.addRefs(inObj);
          }

          @Override
          public boolean isAlive() {
            for (@Nonnull
            final Result element : inObj)
              if (element.isAlive()) {
                return true;
              }
            return false;
          }

          public void _free() {
            ReferenceCounting.freeRefs(inObj);
          }

        };
      } finally {
        ReferenceCounting.freeRefs(inObj);
      }
    } finally {
      if (null != outputA)
        ReferenceCounting.freeRefs(outputA);
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

  public @SuppressWarnings("unused") void _free() {
  }

  public @Override @SuppressWarnings("unused") ProductLayer addRef() {
    return (ProductLayer) super.addRef();
  }
}
