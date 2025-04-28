module     p0_ubaru_httbar_d13h1l1d_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p0_ubaru_httbar/helicity1d13h1l1d_qp.f90
   ! generator: buildfortran_d.py
   use p0_ubaru_httbar_config, only: ki => ki_qp
   use p0_ubaru_httbar_util_qp, only: cond, d => metric_tensor
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   integer, private :: iv0
   integer, private :: iv1
   integer, private :: iv2
   integer, private :: iv3
   real(ki), dimension(4), private :: qshift
   public :: derivative
contains
!---#[ function brack_1:
   pure function brack_1(Q, mu2) result(brack)
      use p0_ubaru_httbar_model_qp
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_color_qp
      use p0_ubaru_httbar_abbrevd13h1_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(15) :: acd13
      complex(ki) :: brack
      acd13(1)=dotproduct(qshift,spvak1k2)
      acd13(2)=dotproduct(qshift,spval4k2)
      acd13(3)=abb13(7)
      acd13(4)=dotproduct(qshift,spval5k2)
      acd13(5)=abb13(9)
      acd13(6)=abb13(8)
      acd13(7)=abb13(12)
      acd13(8)=abb13(13)
      acd13(9)=dotproduct(qshift,spval3k2)
      acd13(10)=abb13(10)
      acd13(11)=abb13(11)
      acd13(12)=acd13(3)*acd13(2)
      acd13(13)=acd13(5)*acd13(4)
      acd13(12)=-acd13(6)+acd13(12)+acd13(13)
      acd13(12)=acd13(1)*acd13(12)
      acd13(13)=-acd13(7)*acd13(2)
      acd13(14)=-acd13(8)*acd13(4)
      acd13(15)=acd13(10)*acd13(9)
      brack=acd13(11)+acd13(12)+acd13(13)+acd13(14)+acd13(15)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p0_ubaru_httbar_model_qp
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_color_qp
      use p0_ubaru_httbar_abbrevd13h1_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(17) :: acd13
      complex(ki) :: brack
      acd13(1)=spvak1k2(iv1)
      acd13(2)=dotproduct(qshift,spval4k2)
      acd13(3)=abb13(7)
      acd13(4)=dotproduct(qshift,spval5k2)
      acd13(5)=abb13(9)
      acd13(6)=abb13(8)
      acd13(7)=spval4k2(iv1)
      acd13(8)=dotproduct(qshift,spvak1k2)
      acd13(9)=abb13(12)
      acd13(10)=spval5k2(iv1)
      acd13(11)=abb13(13)
      acd13(12)=spval3k2(iv1)
      acd13(13)=abb13(10)
      acd13(14)=-acd13(2)*acd13(3)
      acd13(15)=-acd13(4)*acd13(5)
      acd13(14)=acd13(6)+acd13(15)+acd13(14)
      acd13(14)=acd13(1)*acd13(14)
      acd13(15)=-acd13(8)*acd13(3)
      acd13(15)=acd13(9)+acd13(15)
      acd13(15)=acd13(7)*acd13(15)
      acd13(16)=-acd13(8)*acd13(5)
      acd13(16)=acd13(11)+acd13(16)
      acd13(16)=acd13(10)*acd13(16)
      acd13(17)=-acd13(13)*acd13(12)
      brack=acd13(14)+acd13(15)+acd13(16)+acd13(17)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p0_ubaru_httbar_model_qp
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_color_qp
      use p0_ubaru_httbar_abbrevd13h1_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(11) :: acd13
      complex(ki) :: brack
      acd13(1)=spvak1k2(iv1)
      acd13(2)=spval4k2(iv2)
      acd13(3)=abb13(7)
      acd13(4)=spval5k2(iv2)
      acd13(5)=abb13(9)
      acd13(6)=spvak1k2(iv2)
      acd13(7)=spval4k2(iv1)
      acd13(8)=spval5k2(iv1)
      acd13(9)=acd13(2)*acd13(3)
      acd13(10)=acd13(4)*acd13(5)
      acd13(9)=acd13(9)+acd13(10)
      acd13(9)=acd13(1)*acd13(9)
      acd13(10)=acd13(7)*acd13(3)
      acd13(11)=acd13(8)*acd13(5)
      acd13(10)=acd13(11)+acd13(10)
      acd13(10)=acd13(6)*acd13(10)
      brack=acd13(9)+acd13(10)
   end function brack_3
!---#] function brack_3:
!---#[ function brack_4:
   pure function brack_4(Q, mu2) result(brack)
      use p0_ubaru_httbar_model_qp
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_color_qp
      use p0_ubaru_httbar_abbrevd13h1_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(1) :: acd13
      complex(ki) :: brack
      brack=0.0_ki
   end function brack_4
!---#] function brack_4:
!---#[ function derivative:
   function derivative(mu2,i1,i2,i3) result(numerator)
      use p0_ubaru_httbar_globalsl1_qp, only: epspow
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_abbrevd13h1_qp
      implicit none
      complex(ki), intent(in) :: mu2
      integer, intent(in), optional :: i1
      integer, intent(in), optional :: i2
      integer, intent(in), optional :: i3
      complex(ki) :: numerator
      complex(ki) :: loc
      integer :: t1
      integer :: deg
      complex(ki), dimension(4), parameter :: Q = (/ (0.0_ki,0.0_ki),(0.0_ki,0.&
      &0_ki),(0.0_ki,0.0_ki),(0.0_ki,0.0_ki)/)
      qshift = k3+k4
      numerator = 0.0_ki
      deg = 0
      if(present(i1)) then
          iv1=i1
          deg=1
      else
          iv1=1
      end if
      if(present(i2)) then
          iv2=i2
          deg=2
      else
          iv2=1
      end if
      if(present(i3)) then
          iv3=i3
          deg=3
      else
          iv3=1
      end if
      t1 = 0
      if(deg.eq.0) then
         numerator = cond(epspow.eq.t1,brack_1,Q,mu2)
         return
      end if
      if(deg.eq.1) then
         numerator = cond(epspow.eq.t1,brack_2,Q,mu2)
         return
      end if
      if(deg.eq.2) then
         numerator = cond(epspow.eq.t1,brack_3,Q,mu2)
         return
      end if
      if(deg.eq.3) then
         numerator = cond(epspow.eq.t1,brack_4,Q,mu2)
         return
      end if
   end function derivative
!---#] function derivative:
end module     p0_ubaru_httbar_d13h1l1d_qp
