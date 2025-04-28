module     p0_ubaru_httbar_d67h10l1d_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p0_ubaru_httbar/helicity10d67h10l1d_qp.f90
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
      use p0_ubaru_httbar_abbrevd67h10_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(18) :: acd67
      complex(ki) :: brack
      acd67(1)=dotproduct(k2,qshift)
      acd67(2)=dotproduct(qshift,spvak2k1)
      acd67(3)=abb67(9)
      acd67(4)=abb67(8)
      acd67(5)=dotproduct(l4,qshift)
      acd67(6)=abb67(12)
      acd67(7)=dotproduct(qshift,qshift)
      acd67(8)=dotproduct(qshift,spval3k2)
      acd67(9)=abb67(11)
      acd67(10)=abb67(15)
      acd67(11)=abb67(10)
      acd67(12)=dotproduct(qshift,spval3k1)
      acd67(13)=abb67(21)
      acd67(14)=acd67(3)*acd67(1)
      acd67(15)=acd67(9)*acd67(8)
      acd67(14)=-acd67(10)+acd67(15)+acd67(14)
      acd67(14)=acd67(2)*acd67(14)
      acd67(15)=-acd67(7)+acd67(5)
      acd67(15)=acd67(6)*acd67(15)
      acd67(16)=-acd67(4)*acd67(1)
      acd67(17)=-acd67(11)*acd67(8)
      acd67(18)=-acd67(13)*acd67(12)
      brack=acd67(14)+acd67(15)+acd67(16)+acd67(17)+acd67(18)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p0_ubaru_httbar_model_qp
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_color_qp
      use p0_ubaru_httbar_abbrevd67h10_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(21) :: acd67
      complex(ki) :: brack
      acd67(1)=k2(iv1)
      acd67(2)=dotproduct(qshift,spvak2k1)
      acd67(3)=abb67(9)
      acd67(4)=abb67(8)
      acd67(5)=l4(iv1)
      acd67(6)=abb67(12)
      acd67(7)=qshift(iv1)
      acd67(8)=spvak2k1(iv1)
      acd67(9)=dotproduct(k2,qshift)
      acd67(10)=dotproduct(qshift,spval3k2)
      acd67(11)=abb67(11)
      acd67(12)=abb67(15)
      acd67(13)=spval3k2(iv1)
      acd67(14)=abb67(10)
      acd67(15)=spval3k1(iv1)
      acd67(16)=abb67(21)
      acd67(17)=acd67(9)*acd67(3)
      acd67(18)=acd67(10)*acd67(11)
      acd67(17)=-acd67(12)+acd67(18)+acd67(17)
      acd67(17)=acd67(8)*acd67(17)
      acd67(18)=acd67(3)*acd67(2)
      acd67(18)=-acd67(4)+acd67(18)
      acd67(18)=acd67(1)*acd67(18)
      acd67(19)=-2.0_ki*acd67(7)+acd67(5)
      acd67(19)=acd67(6)*acd67(19)
      acd67(20)=acd67(11)*acd67(2)
      acd67(20)=-acd67(14)+acd67(20)
      acd67(20)=acd67(13)*acd67(20)
      acd67(21)=-acd67(16)*acd67(15)
      brack=acd67(17)+acd67(18)+acd67(19)+acd67(20)+acd67(21)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p0_ubaru_httbar_model_qp
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_color_qp
      use p0_ubaru_httbar_abbrevd67h10_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(13) :: acd67
      complex(ki) :: brack
      acd67(1)=d(iv1,iv2)
      acd67(2)=abb67(12)
      acd67(3)=k2(iv1)
      acd67(4)=spvak2k1(iv2)
      acd67(5)=abb67(9)
      acd67(6)=k2(iv2)
      acd67(7)=spvak2k1(iv1)
      acd67(8)=spval3k2(iv2)
      acd67(9)=abb67(11)
      acd67(10)=spval3k2(iv1)
      acd67(11)=acd67(3)*acd67(4)
      acd67(12)=acd67(6)*acd67(7)
      acd67(11)=acd67(12)+acd67(11)
      acd67(11)=acd67(5)*acd67(11)
      acd67(12)=acd67(8)*acd67(7)
      acd67(13)=acd67(10)*acd67(4)
      acd67(12)=acd67(13)+acd67(12)
      acd67(12)=acd67(9)*acd67(12)
      acd67(13)=acd67(2)*acd67(1)
      brack=acd67(11)+acd67(12)-2.0_ki*acd67(13)
   end function brack_3
!---#] function brack_3:
!---#[ function derivative:
   function derivative(mu2,i1,i2) result(numerator)
      use p0_ubaru_httbar_globalsl1_qp, only: epspow
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_abbrevd67h10_qp
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
      qshift = -k2
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
end module     p0_ubaru_httbar_d67h10l1d_qp
