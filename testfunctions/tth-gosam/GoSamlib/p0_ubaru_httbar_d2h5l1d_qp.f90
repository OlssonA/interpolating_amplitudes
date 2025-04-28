module     p0_ubaru_httbar_d2h5l1d_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p0_ubaru_httbar/helicity5d2h5l1d_qp.f90
   ! generator: buildfortran_d.py
   use p0_ubaru_httbar_config, only: ki => ki_qp
   use p0_ubaru_httbar_util_qp, only: cond, d => metric_tensor
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   integer, private :: iv0
   integer, private :: iv1
   integer, private :: iv2
   real(ki), dimension(4), private :: qshift
   public :: derivative
contains
!---#[ function brack_1:
   pure function brack_1(Q, mu2) result(brack)
      use p0_ubaru_httbar_model_qp
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_color_qp
      use p0_ubaru_httbar_abbrevd2h5_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(1) :: acd2
      complex(ki) :: brack
      acd2(1)=abb2(11)
      brack=acd2(1)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p0_ubaru_httbar_model_qp
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_color_qp
      use p0_ubaru_httbar_abbrevd2h5_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(24) :: acd2
      complex(ki) :: brack
      acd2(1)=k1(iv1)
      acd2(2)=abb2(24)
      acd2(3)=k2(iv1)
      acd2(4)=abb2(13)
      acd2(5)=l5(iv1)
      acd2(6)=abb2(21)
      acd2(7)=spvak1k2(iv1)
      acd2(8)=abb2(9)
      acd2(9)=spval3k2(iv1)
      acd2(10)=abb2(14)
      acd2(11)=spval5k2(iv1)
      acd2(12)=abb2(23)
      acd2(13)=spval5l3(iv1)
      acd2(14)=abb2(31)
      acd2(15)=spval5l4(iv1)
      acd2(16)=abb2(28)
      acd2(17)=-acd2(2)*acd2(1)
      acd2(18)=-acd2(4)*acd2(3)
      acd2(19)=-acd2(6)*acd2(5)
      acd2(20)=-acd2(8)*acd2(7)
      acd2(21)=-acd2(10)*acd2(9)
      acd2(22)=-acd2(12)*acd2(11)
      acd2(23)=-acd2(14)*acd2(13)
      acd2(24)=-acd2(16)*acd2(15)
      brack=acd2(17)+acd2(18)+acd2(19)+acd2(20)+acd2(21)+acd2(22)+acd2(23)+acd2&
      &(24)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p0_ubaru_httbar_model_qp
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_color_qp
      use p0_ubaru_httbar_abbrevd2h5_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(21) :: acd2
      complex(ki) :: brack
      acd2(1)=d(iv1,iv2)
      acd2(2)=abb2(21)
      acd2(3)=k2(iv1)
      acd2(4)=spvak1k2(iv2)
      acd2(5)=abb2(10)
      acd2(6)=k2(iv2)
      acd2(7)=spvak1k2(iv1)
      acd2(8)=spval3k2(iv2)
      acd2(9)=abb2(17)
      acd2(10)=spval5l3(iv2)
      acd2(11)=abb2(15)
      acd2(12)=spval5l4(iv2)
      acd2(13)=abb2(12)
      acd2(14)=spval3k2(iv1)
      acd2(15)=spval5l3(iv1)
      acd2(16)=spval5l4(iv1)
      acd2(17)=acd2(3)*acd2(5)
      acd2(18)=acd2(14)*acd2(9)
      acd2(19)=acd2(15)*acd2(11)
      acd2(20)=acd2(16)*acd2(13)
      acd2(17)=acd2(20)+acd2(19)+acd2(18)+acd2(17)
      acd2(17)=acd2(4)*acd2(17)
      acd2(18)=acd2(6)*acd2(5)
      acd2(19)=acd2(8)*acd2(9)
      acd2(20)=acd2(10)*acd2(11)
      acd2(21)=acd2(12)*acd2(13)
      acd2(18)=acd2(21)+acd2(20)+acd2(19)+acd2(18)
      acd2(18)=acd2(7)*acd2(18)
      acd2(19)=acd2(2)*acd2(1)
      brack=acd2(17)+acd2(18)+2.0_ki*acd2(19)
   end function brack_3
!---#] function brack_3:
!---#[ function derivative:
   function derivative(mu2,i1,i2) result(numerator)
      use p0_ubaru_httbar_globalsl1_qp, only: epspow
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_abbrevd2h5_qp
      implicit none
      complex(ki), intent(in) :: mu2
      integer, intent(in), optional :: i1
      integer, intent(in), optional :: i2
      complex(ki) :: numerator
      complex(ki) :: loc
      integer :: t1
      integer :: deg
      complex(ki), dimension(4), parameter :: Q = (/ (0.0_ki,0.0_ki),(0.0_ki,0.&
      &0_ki),(0.0_ki,0.0_ki),(0.0_ki,0.0_ki)/)
      qshift = 0
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
   end function derivative
!---#] function derivative:
end module     p0_ubaru_httbar_d2h5l1d_qp
