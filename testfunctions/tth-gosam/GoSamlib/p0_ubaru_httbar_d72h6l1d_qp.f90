module     p0_ubaru_httbar_d72h6l1d_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p0_ubaru_httbar/helicity6d72h6l1d_qp.f90
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
      use p0_ubaru_httbar_abbrevd72h6_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(31) :: acd72
      complex(ki) :: brack
      acd72(1)=dotproduct(k2,qshift)
      acd72(2)=abb72(13)
      acd72(3)=dotproduct(l3,qshift)
      acd72(4)=abb72(15)
      acd72(5)=dotproduct(l4,qshift)
      acd72(6)=abb72(21)
      acd72(7)=dotproduct(qshift,qshift)
      acd72(8)=abb72(16)
      acd72(9)=dotproduct(qshift,spvak2k1)
      acd72(10)=abb72(12)
      acd72(11)=dotproduct(qshift,spvak2l3)
      acd72(12)=abb72(11)
      acd72(13)=dotproduct(qshift,spvak2l4)
      acd72(14)=abb72(10)
      acd72(15)=dotproduct(qshift,spval3k1)
      acd72(16)=abb72(24)
      acd72(17)=dotproduct(qshift,spval3l4)
      acd72(18)=abb72(25)
      acd72(19)=dotproduct(qshift,spval4l3)
      acd72(20)=abb72(23)
      acd72(21)=abb72(18)
      acd72(22)=-acd72(2)*acd72(1)
      acd72(23)=-acd72(4)*acd72(3)
      acd72(24)=-acd72(6)*acd72(5)
      acd72(25)=acd72(8)*acd72(7)
      acd72(26)=-acd72(10)*acd72(9)
      acd72(27)=-acd72(12)*acd72(11)
      acd72(28)=-acd72(14)*acd72(13)
      acd72(29)=-acd72(16)*acd72(15)
      acd72(30)=-acd72(18)*acd72(17)
      acd72(31)=-acd72(20)*acd72(19)
      brack=acd72(21)+acd72(22)+acd72(23)+acd72(24)+acd72(25)+acd72(26)+acd72(2&
      &7)+acd72(28)+acd72(29)+acd72(30)+acd72(31)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p0_ubaru_httbar_model_qp
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_color_qp
      use p0_ubaru_httbar_abbrevd72h6_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(30) :: acd72
      complex(ki) :: brack
      acd72(1)=k2(iv1)
      acd72(2)=abb72(13)
      acd72(3)=l3(iv1)
      acd72(4)=abb72(15)
      acd72(5)=l4(iv1)
      acd72(6)=abb72(21)
      acd72(7)=qshift(iv1)
      acd72(8)=abb72(16)
      acd72(9)=spvak2k1(iv1)
      acd72(10)=abb72(12)
      acd72(11)=spvak2l3(iv1)
      acd72(12)=abb72(11)
      acd72(13)=spvak2l4(iv1)
      acd72(14)=abb72(10)
      acd72(15)=spval3k1(iv1)
      acd72(16)=abb72(24)
      acd72(17)=spval3l4(iv1)
      acd72(18)=abb72(25)
      acd72(19)=spval4l3(iv1)
      acd72(20)=abb72(23)
      acd72(21)=acd72(2)*acd72(1)
      acd72(22)=acd72(4)*acd72(3)
      acd72(23)=acd72(6)*acd72(5)
      acd72(24)=acd72(8)*acd72(7)
      acd72(25)=acd72(10)*acd72(9)
      acd72(26)=acd72(12)*acd72(11)
      acd72(27)=acd72(14)*acd72(13)
      acd72(28)=acd72(16)*acd72(15)
      acd72(29)=acd72(18)*acd72(17)
      acd72(30)=acd72(20)*acd72(19)
      brack=acd72(21)+acd72(22)+acd72(23)-2.0_ki*acd72(24)+acd72(25)+acd72(26)+&
      &acd72(27)+acd72(28)+acd72(29)+acd72(30)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p0_ubaru_httbar_model_qp
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_color_qp
      use p0_ubaru_httbar_abbrevd72h6_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(3) :: acd72
      complex(ki) :: brack
      acd72(1)=d(iv1,iv2)
      acd72(2)=abb72(16)
      brack=2.0_ki*acd72(2)*acd72(1)
   end function brack_3
!---#] function brack_3:
!---#[ function derivative:
   function derivative(mu2,i1,i2) result(numerator)
      use p0_ubaru_httbar_globalsl1_qp, only: epspow
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_abbrevd72h6_qp
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
      qshift = k3+k5
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
end module     p0_ubaru_httbar_d72h6l1d_qp
