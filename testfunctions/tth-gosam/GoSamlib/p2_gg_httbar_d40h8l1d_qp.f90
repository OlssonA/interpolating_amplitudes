module     p2_gg_httbar_d40h8l1d_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity8d40h8l1d_qp.f90
   ! generator: buildfortran_d.py
   use p2_gg_httbar_config, only: ki => ki_qp
   use p2_gg_httbar_util_qp, only: cond, d => metric_tensor
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
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd40h8_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(1) :: acd40
      complex(ki) :: brack
      acd40(1)=abb40(14)
      brack=acd40(1)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd40h8_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(69) :: acd40
      complex(ki) :: brack
      acd40(1)=k2(iv1)
      acd40(2)=abb40(16)
      acd40(3)=l4(iv1)
      acd40(4)=abb40(32)
      acd40(5)=spvak1l4(iv1)
      acd40(6)=abb40(20)
      acd40(7)=spvak2k1(iv1)
      acd40(8)=abb40(26)
      acd40(9)=spvak2l4(iv1)
      acd40(10)=abb40(23)
      acd40(11)=spvak2l5(iv1)
      acd40(12)=abb40(34)
      acd40(13)=spval4k1(iv1)
      acd40(14)=abb40(30)
      acd40(15)=spval4k2(iv1)
      acd40(16)=abb40(19)
      acd40(17)=spval4l5(iv1)
      acd40(18)=abb40(36)
      acd40(19)=spval5l4(iv1)
      acd40(20)=abb40(35)
      acd40(21)=spvak1e2(iv1)
      acd40(22)=abb40(22)
      acd40(23)=spvae2k1(iv1)
      acd40(24)=abb40(31)
      acd40(25)=spvak2e1(iv1)
      acd40(26)=abb40(15)
      acd40(27)=spvak2e2(iv1)
      acd40(28)=abb40(43)
      acd40(29)=spvae2k2(iv1)
      acd40(30)=abb40(40)
      acd40(31)=spval4e1(iv1)
      acd40(32)=abb40(17)
      acd40(33)=spvae1l4(iv1)
      acd40(34)=abb40(54)
      acd40(35)=spval4e2(iv1)
      acd40(36)=abb40(21)
      acd40(37)=spvae2l4(iv1)
      acd40(38)=abb40(45)
      acd40(39)=spval5e2(iv1)
      acd40(40)=abb40(44)
      acd40(41)=spvae2l5(iv1)
      acd40(42)=abb40(38)
      acd40(43)=spvae1e2(iv1)
      acd40(44)=abb40(24)
      acd40(45)=spvae2e1(iv1)
      acd40(46)=abb40(18)
      acd40(47)=-acd40(2)*acd40(1)
      acd40(48)=-acd40(4)*acd40(3)
      acd40(49)=-acd40(6)*acd40(5)
      acd40(50)=-acd40(8)*acd40(7)
      acd40(51)=-acd40(10)*acd40(9)
      acd40(52)=-acd40(12)*acd40(11)
      acd40(53)=-acd40(14)*acd40(13)
      acd40(54)=-acd40(16)*acd40(15)
      acd40(55)=-acd40(18)*acd40(17)
      acd40(56)=-acd40(20)*acd40(19)
      acd40(57)=-acd40(22)*acd40(21)
      acd40(58)=-acd40(24)*acd40(23)
      acd40(59)=-acd40(26)*acd40(25)
      acd40(60)=-acd40(28)*acd40(27)
      acd40(61)=-acd40(30)*acd40(29)
      acd40(62)=-acd40(32)*acd40(31)
      acd40(63)=-acd40(34)*acd40(33)
      acd40(64)=-acd40(36)*acd40(35)
      acd40(65)=-acd40(38)*acd40(37)
      acd40(66)=-acd40(40)*acd40(39)
      acd40(67)=-acd40(42)*acd40(41)
      acd40(68)=-acd40(44)*acd40(43)
      acd40(69)=-acd40(46)*acd40(45)
      brack=acd40(47)+acd40(48)+acd40(49)+acd40(50)+acd40(51)+acd40(52)+acd40(5&
      &3)+acd40(54)+acd40(55)+acd40(56)+acd40(57)+acd40(58)+acd40(59)+acd40(60)&
      &+acd40(61)+acd40(62)+acd40(63)+acd40(64)+acd40(65)+acd40(66)+acd40(67)+a&
      &cd40(68)+acd40(69)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd40h8_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(1) :: acd40
      complex(ki) :: brack
      brack=0.0_ki
   end function brack_3
!---#] function brack_3:
!---#[ function derivative:
   function derivative(mu2,i1,i2) result(numerator)
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd40h8_qp
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
end module     p2_gg_httbar_d40h8l1d_qp
