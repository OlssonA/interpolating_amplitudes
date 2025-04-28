module     p0_ubaru_httbar_d21h5l1d_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p0_ubaru_httbar/helicity5d21h5l1d_qp.f90
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
      use p0_ubaru_httbar_abbrevd21h5_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(19) :: acd21
      complex(ki) :: brack
      acd21(1)=dotproduct(k2,qshift)
      acd21(2)=abb21(13)
      acd21(3)=dotproduct(l3,qshift)
      acd21(4)=abb21(20)
      acd21(5)=dotproduct(l4,qshift)
      acd21(6)=abb21(22)
      acd21(7)=dotproduct(qshift,spval3k2)
      acd21(8)=abb21(16)
      acd21(9)=dotproduct(qshift,spval3l4)
      acd21(10)=abb21(15)
      acd21(11)=dotproduct(qshift,spval4l3)
      acd21(12)=abb21(14)
      acd21(13)=abb21(11)
      acd21(14)=-acd21(2)*acd21(1)
      acd21(15)=-acd21(4)*acd21(3)
      acd21(16)=-acd21(6)*acd21(5)
      acd21(17)=-acd21(8)*acd21(7)
      acd21(18)=-acd21(10)*acd21(9)
      acd21(19)=-acd21(12)*acd21(11)
      brack=acd21(13)+acd21(14)+acd21(15)+acd21(16)+acd21(17)+acd21(18)+acd21(1&
      &9)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p0_ubaru_httbar_model_qp
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_color_qp
      use p0_ubaru_httbar_abbrevd21h5_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(18) :: acd21
      complex(ki) :: brack
      acd21(1)=k2(iv1)
      acd21(2)=abb21(13)
      acd21(3)=l3(iv1)
      acd21(4)=abb21(20)
      acd21(5)=l4(iv1)
      acd21(6)=abb21(22)
      acd21(7)=spval3k2(iv1)
      acd21(8)=abb21(16)
      acd21(9)=spval3l4(iv1)
      acd21(10)=abb21(15)
      acd21(11)=spval4l3(iv1)
      acd21(12)=abb21(14)
      acd21(13)=acd21(2)*acd21(1)
      acd21(14)=acd21(4)*acd21(3)
      acd21(15)=acd21(6)*acd21(5)
      acd21(16)=acd21(8)*acd21(7)
      acd21(17)=acd21(10)*acd21(9)
      acd21(18)=acd21(12)*acd21(11)
      brack=acd21(13)+acd21(14)+acd21(15)+acd21(16)+acd21(17)+acd21(18)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p0_ubaru_httbar_model_qp
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_color_qp
      use p0_ubaru_httbar_abbrevd21h5_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(1) :: acd21
      complex(ki) :: brack
      brack=0.0_ki
   end function brack_3
!---#] function brack_3:
!---#[ function derivative:
   function derivative(mu2,i1,i2) result(numerator)
      use p0_ubaru_httbar_globalsl1_qp, only: epspow
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_abbrevd21h5_qp
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
      qshift = k3+k4+k5
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
end module     p0_ubaru_httbar_d21h5l1d_qp
