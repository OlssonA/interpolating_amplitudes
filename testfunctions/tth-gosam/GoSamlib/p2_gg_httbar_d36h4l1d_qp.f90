module     p2_gg_httbar_d36h4l1d_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity4d36h4l1d_qp.f90
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
      use p2_gg_httbar_abbrevd36h4_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(1) :: acd36
      complex(ki) :: brack
      acd36(1)=abb36(28)
      brack=acd36(1)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd36h4_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(60) :: acd36
      complex(ki) :: brack
      acd36(1)=k2(iv1)
      acd36(2)=abb36(16)
      acd36(3)=l3(iv1)
      acd36(4)=abb36(29)
      acd36(5)=l4(iv1)
      acd36(6)=abb36(98)
      acd36(7)=spvak1l3(iv1)
      acd36(8)=abb36(26)
      acd36(9)=spvak1l4(iv1)
      acd36(10)=abb36(25)
      acd36(11)=spvak2k1(iv1)
      acd36(12)=abb36(19)
      acd36(13)=spvak2l3(iv1)
      acd36(14)=abb36(18)
      acd36(15)=spvak2l4(iv1)
      acd36(16)=abb36(15)
      acd36(17)=spval3k1(iv1)
      acd36(18)=abb36(39)
      acd36(19)=spval3k2(iv1)
      acd36(20)=abb36(23)
      acd36(21)=spval3l4(iv1)
      acd36(22)=abb36(21)
      acd36(23)=spval4l3(iv1)
      acd36(24)=abb36(22)
      acd36(25)=spvak2e1(iv1)
      acd36(26)=abb36(20)
      acd36(27)=spvak2e2(iv1)
      acd36(28)=abb36(27)
      acd36(29)=spval3e1(iv1)
      acd36(30)=abb36(40)
      acd36(31)=spvae1l3(iv1)
      acd36(32)=abb36(38)
      acd36(33)=spval3e2(iv1)
      acd36(34)=abb36(35)
      acd36(35)=spvae2l3(iv1)
      acd36(36)=abb36(33)
      acd36(37)=spvae1l4(iv1)
      acd36(38)=abb36(30)
      acd36(39)=spvae2l4(iv1)
      acd36(40)=abb36(24)
      acd36(41)=-acd36(2)*acd36(1)
      acd36(42)=-acd36(4)*acd36(3)
      acd36(43)=-acd36(6)*acd36(5)
      acd36(44)=-acd36(8)*acd36(7)
      acd36(45)=-acd36(10)*acd36(9)
      acd36(46)=-acd36(12)*acd36(11)
      acd36(47)=-acd36(14)*acd36(13)
      acd36(48)=-acd36(16)*acd36(15)
      acd36(49)=-acd36(18)*acd36(17)
      acd36(50)=-acd36(20)*acd36(19)
      acd36(51)=-acd36(22)*acd36(21)
      acd36(52)=-acd36(24)*acd36(23)
      acd36(53)=-acd36(26)*acd36(25)
      acd36(54)=-acd36(28)*acd36(27)
      acd36(55)=-acd36(30)*acd36(29)
      acd36(56)=-acd36(32)*acd36(31)
      acd36(57)=-acd36(34)*acd36(33)
      acd36(58)=-acd36(36)*acd36(35)
      acd36(59)=-acd36(38)*acd36(37)
      acd36(60)=-acd36(40)*acd36(39)
      brack=acd36(41)+acd36(42)+acd36(43)+acd36(44)+acd36(45)+acd36(46)+acd36(4&
      &7)+acd36(48)+acd36(49)+acd36(50)+acd36(51)+acd36(52)+acd36(53)+acd36(54)&
      &+acd36(55)+acd36(56)+acd36(57)+acd36(58)+acd36(59)+acd36(60)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd36h4_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(1) :: acd36
      complex(ki) :: brack
      brack=0.0_ki
   end function brack_3
!---#] function brack_3:
!---#[ function derivative:
   function derivative(mu2,i1,i2) result(numerator)
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd36h4_qp
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
end module     p2_gg_httbar_d36h4l1d_qp
