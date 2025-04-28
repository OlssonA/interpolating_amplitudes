module     p0_ubaru_httbar_d66h13l1d_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p0_ubaru_httbar/helicity13d66h13l1d_qp.f90
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
      use p0_ubaru_httbar_abbrevd66h13_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(22) :: acd66
      complex(ki) :: brack
      acd66(1)=dotproduct(k1,qshift)
      acd66(2)=abb66(10)
      acd66(3)=dotproduct(k2,qshift)
      acd66(4)=abb66(12)
      acd66(5)=dotproduct(qshift,qshift)
      acd66(6)=dotproduct(qshift,spvak1k2)
      acd66(7)=dotproduct(qshift,spvak2l4)
      acd66(8)=abb66(9)
      acd66(9)=dotproduct(qshift,spval3l4)
      acd66(10)=abb66(16)
      acd66(11)=abb66(13)
      acd66(12)=abb66(15)
      acd66(13)=dotproduct(qshift,spvak1l4)
      acd66(14)=abb66(14)
      acd66(15)=dotproduct(qshift,spval3k2)
      acd66(16)=abb66(20)
      acd66(17)=acd66(5)+acd66(1)
      acd66(17)=acd66(2)*acd66(17)
      acd66(18)=acd66(8)*acd66(6)
      acd66(18)=-acd66(11)+acd66(18)
      acd66(18)=acd66(7)*acd66(18)
      acd66(19)=acd66(10)*acd66(6)
      acd66(19)=-acd66(12)+acd66(19)
      acd66(19)=acd66(9)*acd66(19)
      acd66(20)=-acd66(4)*acd66(3)
      acd66(21)=-acd66(14)*acd66(13)
      acd66(22)=-acd66(16)*acd66(15)
      brack=acd66(17)+acd66(18)+acd66(19)+acd66(20)+acd66(21)+acd66(22)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p0_ubaru_httbar_model_qp
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_color_qp
      use p0_ubaru_httbar_abbrevd66h13_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(26) :: acd66
      complex(ki) :: brack
      acd66(1)=k1(iv1)
      acd66(2)=abb66(10)
      acd66(3)=k2(iv1)
      acd66(4)=abb66(12)
      acd66(5)=qshift(iv1)
      acd66(6)=spvak1k2(iv1)
      acd66(7)=dotproduct(qshift,spvak2l4)
      acd66(8)=abb66(9)
      acd66(9)=dotproduct(qshift,spval3l4)
      acd66(10)=abb66(16)
      acd66(11)=spvak2l4(iv1)
      acd66(12)=dotproduct(qshift,spvak1k2)
      acd66(13)=abb66(13)
      acd66(14)=spval3l4(iv1)
      acd66(15)=abb66(15)
      acd66(16)=spvak1l4(iv1)
      acd66(17)=abb66(14)
      acd66(18)=spval3k2(iv1)
      acd66(19)=abb66(20)
      acd66(20)=-acd66(7)*acd66(8)
      acd66(21)=-acd66(9)*acd66(10)
      acd66(20)=acd66(21)+acd66(20)
      acd66(20)=acd66(6)*acd66(20)
      acd66(21)=-2.0_ki*acd66(5)-acd66(1)
      acd66(21)=acd66(2)*acd66(21)
      acd66(22)=-acd66(12)*acd66(8)
      acd66(22)=acd66(13)+acd66(22)
      acd66(22)=acd66(11)*acd66(22)
      acd66(23)=-acd66(12)*acd66(10)
      acd66(23)=acd66(15)+acd66(23)
      acd66(23)=acd66(14)*acd66(23)
      acd66(24)=acd66(4)*acd66(3)
      acd66(25)=acd66(17)*acd66(16)
      acd66(26)=acd66(19)*acd66(18)
      brack=acd66(20)+acd66(21)+acd66(22)+acd66(23)+acd66(24)+acd66(25)+acd66(2&
      &6)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p0_ubaru_httbar_model_qp
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_color_qp
      use p0_ubaru_httbar_abbrevd66h13_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(13) :: acd66
      complex(ki) :: brack
      acd66(1)=d(iv1,iv2)
      acd66(2)=abb66(10)
      acd66(3)=spvak1k2(iv1)
      acd66(4)=spvak2l4(iv2)
      acd66(5)=abb66(9)
      acd66(6)=spval3l4(iv2)
      acd66(7)=abb66(16)
      acd66(8)=spvak1k2(iv2)
      acd66(9)=spvak2l4(iv1)
      acd66(10)=spval3l4(iv1)
      acd66(11)=acd66(4)*acd66(5)
      acd66(12)=acd66(6)*acd66(7)
      acd66(11)=acd66(12)+acd66(11)
      acd66(11)=acd66(3)*acd66(11)
      acd66(12)=acd66(9)*acd66(5)
      acd66(13)=acd66(10)*acd66(7)
      acd66(12)=acd66(13)+acd66(12)
      acd66(12)=acd66(8)*acd66(12)
      acd66(13)=acd66(2)*acd66(1)
      brack=acd66(11)+acd66(12)+2.0_ki*acd66(13)
   end function brack_3
!---#] function brack_3:
!---#[ function derivative:
   function derivative(mu2,i1,i2) result(numerator)
      use p0_ubaru_httbar_globalsl1_qp, only: epspow
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_abbrevd66h13_qp
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
      qshift = k2
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
end module     p0_ubaru_httbar_d66h13l1d_qp
