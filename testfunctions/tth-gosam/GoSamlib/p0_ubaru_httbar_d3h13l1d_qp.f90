module     p0_ubaru_httbar_d3h13l1d_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p0_ubaru_httbar/helicity13d3h13l1d_qp.f90
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
      use p0_ubaru_httbar_abbrevd3h13_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(20) :: acd3
      complex(ki) :: brack
      acd3(1)=dotproduct(k2,qshift)
      acd3(2)=dotproduct(qshift,spvak1l3)
      acd3(3)=abb3(15)
      acd3(4)=dotproduct(qshift,spvak1l4)
      acd3(5)=abb3(12)
      acd3(6)=dotproduct(qshift,spvak1l5)
      acd3(7)=abb3(13)
      acd3(8)=abb3(22)
      acd3(9)=abb3(11)
      acd3(10)=dotproduct(qshift,spval3k2)
      acd3(11)=abb3(14)
      acd3(12)=abb3(10)
      acd3(13)=abb3(9)
      acd3(14)=abb3(23)
      acd3(15)=abb3(18)
      acd3(16)=acd3(3)*acd3(2)
      acd3(17)=acd3(5)*acd3(4)
      acd3(18)=acd3(7)*acd3(6)
      acd3(16)=-acd3(8)+acd3(18)+acd3(16)+acd3(17)
      acd3(16)=acd3(1)*acd3(16)
      acd3(17)=acd3(11)*acd3(4)
      acd3(17)=-acd3(14)+acd3(17)
      acd3(17)=acd3(10)*acd3(17)
      acd3(18)=-acd3(9)*acd3(2)
      acd3(19)=-acd3(12)*acd3(4)
      acd3(20)=-acd3(13)*acd3(6)
      brack=acd3(15)+acd3(16)+acd3(17)+acd3(18)+acd3(19)+acd3(20)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p0_ubaru_httbar_model_qp
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_color_qp
      use p0_ubaru_httbar_abbrevd3h13_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(25) :: acd3
      complex(ki) :: brack
      acd3(1)=k2(iv1)
      acd3(2)=dotproduct(qshift,spvak1l3)
      acd3(3)=abb3(15)
      acd3(4)=dotproduct(qshift,spvak1l4)
      acd3(5)=abb3(12)
      acd3(6)=dotproduct(qshift,spvak1l5)
      acd3(7)=abb3(13)
      acd3(8)=abb3(22)
      acd3(9)=spvak1l3(iv1)
      acd3(10)=dotproduct(k2,qshift)
      acd3(11)=abb3(11)
      acd3(12)=spvak1l4(iv1)
      acd3(13)=dotproduct(qshift,spval3k2)
      acd3(14)=abb3(14)
      acd3(15)=abb3(10)
      acd3(16)=spvak1l5(iv1)
      acd3(17)=abb3(9)
      acd3(18)=spval3k2(iv1)
      acd3(19)=abb3(23)
      acd3(20)=-acd3(2)*acd3(3)
      acd3(21)=-acd3(6)*acd3(7)
      acd3(20)=acd3(8)+acd3(21)+acd3(20)
      acd3(20)=acd3(1)*acd3(20)
      acd3(21)=-acd3(12)*acd3(10)
      acd3(22)=-acd3(4)*acd3(1)
      acd3(21)=acd3(21)+acd3(22)
      acd3(21)=acd3(5)*acd3(21)
      acd3(22)=-acd3(13)*acd3(14)
      acd3(22)=acd3(15)+acd3(22)
      acd3(22)=acd3(12)*acd3(22)
      acd3(23)=-acd3(3)*acd3(10)
      acd3(23)=acd3(11)+acd3(23)
      acd3(23)=acd3(9)*acd3(23)
      acd3(24)=-acd3(7)*acd3(10)
      acd3(24)=acd3(17)+acd3(24)
      acd3(24)=acd3(16)*acd3(24)
      acd3(25)=-acd3(14)*acd3(4)
      acd3(25)=acd3(19)+acd3(25)
      acd3(25)=acd3(18)*acd3(25)
      brack=acd3(20)+acd3(21)+acd3(22)+acd3(23)+acd3(24)+acd3(25)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p0_ubaru_httbar_model_qp
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_color_qp
      use p0_ubaru_httbar_abbrevd3h13_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(18) :: acd3
      complex(ki) :: brack
      acd3(1)=k2(iv1)
      acd3(2)=spvak1l3(iv2)
      acd3(3)=abb3(15)
      acd3(4)=spvak1l4(iv2)
      acd3(5)=abb3(12)
      acd3(6)=spvak1l5(iv2)
      acd3(7)=abb3(13)
      acd3(8)=k2(iv2)
      acd3(9)=spvak1l3(iv1)
      acd3(10)=spvak1l4(iv1)
      acd3(11)=spvak1l5(iv1)
      acd3(12)=spval3k2(iv2)
      acd3(13)=abb3(14)
      acd3(14)=spval3k2(iv1)
      acd3(15)=acd3(5)*acd3(4)
      acd3(16)=acd3(2)*acd3(3)
      acd3(17)=acd3(6)*acd3(7)
      acd3(15)=acd3(17)+acd3(16)+acd3(15)
      acd3(15)=acd3(1)*acd3(15)
      acd3(16)=acd3(10)*acd3(5)
      acd3(17)=acd3(9)*acd3(3)
      acd3(18)=acd3(11)*acd3(7)
      acd3(16)=acd3(18)+acd3(17)+acd3(16)
      acd3(16)=acd3(8)*acd3(16)
      acd3(17)=acd3(12)*acd3(10)
      acd3(18)=acd3(14)*acd3(4)
      acd3(17)=acd3(18)+acd3(17)
      acd3(17)=acd3(13)*acd3(17)
      brack=acd3(15)+acd3(16)+acd3(17)
   end function brack_3
!---#] function brack_3:
!---#[ function derivative:
   function derivative(mu2,i1,i2) result(numerator)
      use p0_ubaru_httbar_globalsl1_qp, only: epspow
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_abbrevd3h13_qp
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
      qshift = k4
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
end module     p0_ubaru_httbar_d3h13l1d_qp
