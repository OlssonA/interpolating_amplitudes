module     p2_gg_httbar_d38h0l1d_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity0d38h0l1d_qp.f90
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
      use p2_gg_httbar_abbrevd38h0_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(1) :: acd38
      complex(ki) :: brack
      acd38(1)=abb38(26)
      brack=acd38(1)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd38h0_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(69) :: acd38
      complex(ki) :: brack
      acd38(1)=k2(iv1)
      acd38(2)=abb38(16)
      acd38(3)=l5(iv1)
      acd38(4)=abb38(28)
      acd38(5)=spvak1l5(iv1)
      acd38(6)=abb38(18)
      acd38(7)=spvak2k1(iv1)
      acd38(8)=abb38(24)
      acd38(9)=spvak2l4(iv1)
      acd38(10)=abb38(22)
      acd38(11)=spvak2l5(iv1)
      acd38(12)=abb38(15)
      acd38(13)=spval4l5(iv1)
      acd38(14)=abb38(41)
      acd38(15)=spval5k1(iv1)
      acd38(16)=abb38(38)
      acd38(17)=spval5k2(iv1)
      acd38(18)=abb38(37)
      acd38(19)=spval5l4(iv1)
      acd38(20)=abb38(32)
      acd38(21)=spvak1e2(iv1)
      acd38(22)=abb38(20)
      acd38(23)=spvae2k1(iv1)
      acd38(24)=abb38(30)
      acd38(25)=spvak2e1(iv1)
      acd38(26)=abb38(23)
      acd38(27)=spvak2e2(iv1)
      acd38(28)=abb38(19)
      acd38(29)=spvae2k2(iv1)
      acd38(30)=abb38(14)
      acd38(31)=spval4e2(iv1)
      acd38(32)=abb38(44)
      acd38(33)=spvae2l4(iv1)
      acd38(34)=abb38(34)
      acd38(35)=spval5e1(iv1)
      acd38(36)=abb38(33)
      acd38(37)=spvae1l5(iv1)
      acd38(38)=abb38(21)
      acd38(39)=spval5e2(iv1)
      acd38(40)=abb38(48)
      acd38(41)=spvae2l5(iv1)
      acd38(42)=abb38(17)
      acd38(43)=spvae1e2(iv1)
      acd38(44)=abb38(27)
      acd38(45)=spvae2e1(iv1)
      acd38(46)=abb38(31)
      acd38(47)=acd38(2)*acd38(1)
      acd38(48)=acd38(4)*acd38(3)
      acd38(49)=acd38(6)*acd38(5)
      acd38(50)=acd38(8)*acd38(7)
      acd38(51)=acd38(10)*acd38(9)
      acd38(52)=acd38(12)*acd38(11)
      acd38(53)=acd38(14)*acd38(13)
      acd38(54)=acd38(16)*acd38(15)
      acd38(55)=acd38(18)*acd38(17)
      acd38(56)=acd38(20)*acd38(19)
      acd38(57)=acd38(22)*acd38(21)
      acd38(58)=acd38(24)*acd38(23)
      acd38(59)=acd38(26)*acd38(25)
      acd38(60)=acd38(28)*acd38(27)
      acd38(61)=acd38(30)*acd38(29)
      acd38(62)=acd38(32)*acd38(31)
      acd38(63)=acd38(34)*acd38(33)
      acd38(64)=acd38(36)*acd38(35)
      acd38(65)=acd38(38)*acd38(37)
      acd38(66)=acd38(40)*acd38(39)
      acd38(67)=acd38(42)*acd38(41)
      acd38(68)=acd38(44)*acd38(43)
      acd38(69)=acd38(46)*acd38(45)
      brack=acd38(47)+acd38(48)+acd38(49)+acd38(50)+acd38(51)+acd38(52)+acd38(5&
      &3)+acd38(54)+acd38(55)+acd38(56)+acd38(57)+acd38(58)+acd38(59)+acd38(60)&
      &+acd38(61)+acd38(62)+acd38(63)+acd38(64)+acd38(65)+acd38(66)+acd38(67)+a&
      &cd38(68)+acd38(69)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd38h0_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(1) :: acd38
      complex(ki) :: brack
      brack=0.0_ki
   end function brack_3
!---#] function brack_3:
!---#[ function derivative:
   function derivative(mu2,i1,i2) result(numerator)
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd38h0_qp
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
end module     p2_gg_httbar_d38h0l1d_qp
