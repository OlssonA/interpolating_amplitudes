module     p2_gg_httbar_d130h12l1d_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity12d130h12l1d_qp.f90
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
      use p2_gg_httbar_abbrevd130h12_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(64) :: acd130
      complex(ki) :: brack
      acd130(1)=dotproduct(k2,qshift)
      acd130(2)=abb130(25)
      acd130(3)=dotproduct(l3,qshift)
      acd130(4)=abb130(23)
      acd130(5)=dotproduct(l4,qshift)
      acd130(6)=abb130(21)
      acd130(7)=dotproduct(qshift,qshift)
      acd130(8)=abb130(22)
      acd130(9)=dotproduct(qshift,spvak1l3)
      acd130(10)=abb130(17)
      acd130(11)=dotproduct(qshift,spvak1l4)
      acd130(12)=abb130(16)
      acd130(13)=dotproduct(qshift,spvak2k1)
      acd130(14)=abb130(15)
      acd130(15)=dotproduct(qshift,spvak2l3)
      acd130(16)=abb130(13)
      acd130(17)=dotproduct(qshift,spvak2l4)
      acd130(18)=abb130(14)
      acd130(19)=dotproduct(qshift,spval3k1)
      acd130(20)=abb130(32)
      acd130(21)=dotproduct(qshift,spval3k2)
      acd130(22)=abb130(31)
      acd130(23)=dotproduct(qshift,spval3l4)
      acd130(24)=abb130(54)
      acd130(25)=dotproduct(qshift,spval4l3)
      acd130(26)=abb130(60)
      acd130(27)=dotproduct(qshift,spvak2e1)
      acd130(28)=abb130(27)
      acd130(29)=dotproduct(qshift,spvak2e2)
      acd130(30)=abb130(44)
      acd130(31)=dotproduct(qshift,spval3e1)
      acd130(32)=abb130(179)
      acd130(33)=dotproduct(qshift,spvae1l3)
      acd130(34)=abb130(68)
      acd130(35)=dotproduct(qshift,spval3e2)
      acd130(36)=abb130(67)
      acd130(37)=dotproduct(qshift,spvae2l3)
      acd130(38)=abb130(48)
      acd130(39)=dotproduct(qshift,spvae1l4)
      acd130(40)=abb130(26)
      acd130(41)=dotproduct(qshift,spvae2l4)
      acd130(42)=abb130(40)
      acd130(43)=abb130(18)
      acd130(44)=-acd130(2)*acd130(1)
      acd130(45)=-acd130(4)*acd130(3)
      acd130(46)=-acd130(6)*acd130(5)
      acd130(47)=acd130(8)*acd130(7)
      acd130(48)=-acd130(10)*acd130(9)
      acd130(49)=-acd130(12)*acd130(11)
      acd130(50)=-acd130(14)*acd130(13)
      acd130(51)=-acd130(16)*acd130(15)
      acd130(52)=-acd130(18)*acd130(17)
      acd130(53)=-acd130(20)*acd130(19)
      acd130(54)=-acd130(22)*acd130(21)
      acd130(55)=-acd130(24)*acd130(23)
      acd130(56)=-acd130(26)*acd130(25)
      acd130(57)=-acd130(28)*acd130(27)
      acd130(58)=-acd130(30)*acd130(29)
      acd130(59)=acd130(32)*acd130(31)
      acd130(60)=-acd130(34)*acd130(33)
      acd130(61)=-acd130(36)*acd130(35)
      acd130(62)=-acd130(38)*acd130(37)
      acd130(63)=-acd130(40)*acd130(39)
      acd130(64)=-acd130(42)*acd130(41)
      brack=acd130(43)+acd130(44)+acd130(45)+acd130(46)+acd130(47)+acd130(48)+a&
      &cd130(49)+acd130(50)+acd130(51)+acd130(52)+acd130(53)+acd130(54)+acd130(&
      &55)+acd130(56)+acd130(57)+acd130(58)+acd130(59)+acd130(60)+acd130(61)+ac&
      &d130(62)+acd130(63)+acd130(64)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd130h12_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(63) :: acd130
      complex(ki) :: brack
      acd130(1)=k2(iv1)
      acd130(2)=abb130(25)
      acd130(3)=l3(iv1)
      acd130(4)=abb130(23)
      acd130(5)=l4(iv1)
      acd130(6)=abb130(21)
      acd130(7)=qshift(iv1)
      acd130(8)=abb130(22)
      acd130(9)=spvak1l3(iv1)
      acd130(10)=abb130(17)
      acd130(11)=spvak1l4(iv1)
      acd130(12)=abb130(16)
      acd130(13)=spvak2k1(iv1)
      acd130(14)=abb130(15)
      acd130(15)=spvak2l3(iv1)
      acd130(16)=abb130(13)
      acd130(17)=spvak2l4(iv1)
      acd130(18)=abb130(14)
      acd130(19)=spval3k1(iv1)
      acd130(20)=abb130(32)
      acd130(21)=spval3k2(iv1)
      acd130(22)=abb130(31)
      acd130(23)=spval3l4(iv1)
      acd130(24)=abb130(54)
      acd130(25)=spval4l3(iv1)
      acd130(26)=abb130(60)
      acd130(27)=spvak2e1(iv1)
      acd130(28)=abb130(27)
      acd130(29)=spvak2e2(iv1)
      acd130(30)=abb130(44)
      acd130(31)=spval3e1(iv1)
      acd130(32)=abb130(179)
      acd130(33)=spvae1l3(iv1)
      acd130(34)=abb130(68)
      acd130(35)=spval3e2(iv1)
      acd130(36)=abb130(67)
      acd130(37)=spvae2l3(iv1)
      acd130(38)=abb130(48)
      acd130(39)=spvae1l4(iv1)
      acd130(40)=abb130(26)
      acd130(41)=spvae2l4(iv1)
      acd130(42)=abb130(40)
      acd130(43)=-acd130(2)*acd130(1)
      acd130(44)=-acd130(4)*acd130(3)
      acd130(45)=-acd130(6)*acd130(5)
      acd130(46)=acd130(8)*acd130(7)
      acd130(47)=-acd130(10)*acd130(9)
      acd130(48)=-acd130(12)*acd130(11)
      acd130(49)=-acd130(14)*acd130(13)
      acd130(50)=-acd130(16)*acd130(15)
      acd130(51)=-acd130(18)*acd130(17)
      acd130(52)=-acd130(20)*acd130(19)
      acd130(53)=-acd130(22)*acd130(21)
      acd130(54)=-acd130(24)*acd130(23)
      acd130(55)=-acd130(26)*acd130(25)
      acd130(56)=-acd130(28)*acd130(27)
      acd130(57)=-acd130(30)*acd130(29)
      acd130(58)=acd130(32)*acd130(31)
      acd130(59)=-acd130(34)*acd130(33)
      acd130(60)=-acd130(36)*acd130(35)
      acd130(61)=-acd130(38)*acd130(37)
      acd130(62)=-acd130(40)*acd130(39)
      acd130(63)=-acd130(42)*acd130(41)
      brack=acd130(43)+acd130(44)+acd130(45)+2.0_ki*acd130(46)+acd130(47)+acd13&
      &0(48)+acd130(49)+acd130(50)+acd130(51)+acd130(52)+acd130(53)+acd130(54)+&
      &acd130(55)+acd130(56)+acd130(57)+acd130(58)+acd130(59)+acd130(60)+acd130&
      &(61)+acd130(62)+acd130(63)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd130h12_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(3) :: acd130
      complex(ki) :: brack
      acd130(1)=d(iv1,iv2)
      acd130(2)=abb130(22)
      brack=2.0_ki*acd130(2)*acd130(1)
   end function brack_3
!---#] function brack_3:
!---#[ function derivative:
   function derivative(mu2,i1,i2) result(numerator)
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd130h12_qp
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
      qshift = -k4
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
end module     p2_gg_httbar_d130h12l1d_qp
