module     p2_gg_httbar_d132h4l1d
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity4d132h4l1d.f90
   ! generator: buildfortran_d.py
   use p2_gg_httbar_config, only: ki
   use p2_gg_httbar_util, only: cond, d => metric_tensor
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
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd132h4
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(40) :: acd132
      complex(ki) :: brack
      acd132(1)=dotproduct(qshift,spvak1e2)
      acd132(2)=dotproduct(qshift,spvae2k2)
      acd132(3)=abb132(59)
      acd132(4)=abb132(19)
      acd132(5)=dotproduct(qshift,spval5e2)
      acd132(6)=abb132(20)
      acd132(7)=dotproduct(qshift,spvak2e2)
      acd132(8)=abb132(12)
      acd132(9)=dotproduct(qshift,spval4e2)
      acd132(10)=abb132(61)
      acd132(11)=dotproduct(qshift,spvae1e2)
      acd132(12)=abb132(53)
      acd132(13)=abb132(17)
      acd132(14)=dotproduct(qshift,spvae2k1)
      acd132(15)=abb132(13)
      acd132(16)=abb132(16)
      acd132(17)=dotproduct(qshift,spvae2l4)
      acd132(18)=abb132(31)
      acd132(19)=dotproduct(qshift,spvae2l5)
      acd132(20)=abb132(22)
      acd132(21)=dotproduct(qshift,spvae2e1)
      acd132(22)=abb132(14)
      acd132(23)=abb132(29)
      acd132(24)=abb132(21)
      acd132(25)=abb132(48)
      acd132(26)=abb132(25)
      acd132(27)=abb132(28)
      acd132(28)=abb132(27)
      acd132(29)=abb132(18)
      acd132(30)=abb132(15)
      acd132(31)=acd132(6)*acd132(2)
      acd132(32)=acd132(15)*acd132(14)
      acd132(33)=acd132(18)*acd132(17)
      acd132(34)=acd132(20)*acd132(19)
      acd132(35)=acd132(22)*acd132(21)
      acd132(31)=-acd132(23)+acd132(35)+acd132(34)+acd132(33)+acd132(32)+acd132&
      &(31)
      acd132(31)=acd132(5)*acd132(31)
      acd132(32)=acd132(3)*acd132(1)
      acd132(33)=acd132(8)*acd132(7)
      acd132(34)=-acd132(10)*acd132(9)
      acd132(35)=-acd132(12)*acd132(11)
      acd132(32)=-acd132(13)+acd132(35)+acd132(34)+acd132(33)+acd132(32)
      acd132(32)=acd132(2)*acd132(32)
      acd132(33)=-acd132(4)*acd132(1)
      acd132(34)=-acd132(16)*acd132(14)
      acd132(35)=-acd132(24)*acd132(7)
      acd132(36)=-acd132(25)*acd132(9)
      acd132(37)=-acd132(26)*acd132(11)
      acd132(38)=-acd132(27)*acd132(17)
      acd132(39)=-acd132(28)*acd132(19)
      acd132(40)=-acd132(29)*acd132(21)
      brack=acd132(30)+acd132(31)+acd132(32)+acd132(33)+acd132(34)+acd132(35)+a&
      &cd132(36)+acd132(37)+acd132(38)+acd132(39)+acd132(40)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd132h4
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(51) :: acd132
      complex(ki) :: brack
      acd132(1)=spvak1e2(iv1)
      acd132(2)=dotproduct(qshift,spvae2k2)
      acd132(3)=abb132(59)
      acd132(4)=abb132(19)
      acd132(5)=spvae2k2(iv1)
      acd132(6)=dotproduct(qshift,spvak1e2)
      acd132(7)=dotproduct(qshift,spval5e2)
      acd132(8)=abb132(20)
      acd132(9)=dotproduct(qshift,spvak2e2)
      acd132(10)=abb132(12)
      acd132(11)=dotproduct(qshift,spval4e2)
      acd132(12)=abb132(61)
      acd132(13)=dotproduct(qshift,spvae1e2)
      acd132(14)=abb132(53)
      acd132(15)=abb132(17)
      acd132(16)=spvae2k1(iv1)
      acd132(17)=abb132(13)
      acd132(18)=abb132(16)
      acd132(19)=spval5e2(iv1)
      acd132(20)=dotproduct(qshift,spvae2k1)
      acd132(21)=dotproduct(qshift,spvae2l4)
      acd132(22)=abb132(31)
      acd132(23)=dotproduct(qshift,spvae2l5)
      acd132(24)=abb132(22)
      acd132(25)=dotproduct(qshift,spvae2e1)
      acd132(26)=abb132(14)
      acd132(27)=abb132(29)
      acd132(28)=spvak2e2(iv1)
      acd132(29)=abb132(21)
      acd132(30)=spval4e2(iv1)
      acd132(31)=abb132(48)
      acd132(32)=spvae1e2(iv1)
      acd132(33)=abb132(25)
      acd132(34)=spvae2l4(iv1)
      acd132(35)=abb132(28)
      acd132(36)=spvae2l5(iv1)
      acd132(37)=abb132(27)
      acd132(38)=spvae2e1(iv1)
      acd132(39)=abb132(18)
      acd132(40)=-acd132(26)*acd132(25)
      acd132(41)=-acd132(24)*acd132(23)
      acd132(42)=-acd132(22)*acd132(21)
      acd132(43)=-acd132(17)*acd132(20)
      acd132(44)=-acd132(2)*acd132(8)
      acd132(40)=acd132(44)+acd132(43)+acd132(42)+acd132(41)+acd132(27)+acd132(&
      &40)
      acd132(40)=acd132(19)*acd132(40)
      acd132(41)=acd132(14)*acd132(13)
      acd132(42)=acd132(12)*acd132(11)
      acd132(43)=-acd132(10)*acd132(9)
      acd132(44)=-acd132(3)*acd132(6)
      acd132(45)=-acd132(7)*acd132(8)
      acd132(41)=acd132(45)+acd132(44)+acd132(43)+acd132(42)+acd132(15)+acd132(&
      &41)
      acd132(41)=acd132(5)*acd132(41)
      acd132(42)=-acd132(26)*acd132(38)
      acd132(43)=-acd132(24)*acd132(36)
      acd132(44)=-acd132(22)*acd132(34)
      acd132(45)=-acd132(16)*acd132(17)
      acd132(42)=acd132(45)+acd132(44)+acd132(42)+acd132(43)
      acd132(42)=acd132(7)*acd132(42)
      acd132(43)=acd132(14)*acd132(32)
      acd132(44)=acd132(12)*acd132(30)
      acd132(45)=-acd132(10)*acd132(28)
      acd132(46)=-acd132(1)*acd132(3)
      acd132(43)=acd132(46)+acd132(45)+acd132(43)+acd132(44)
      acd132(43)=acd132(2)*acd132(43)
      acd132(44)=acd132(38)*acd132(39)
      acd132(45)=acd132(36)*acd132(37)
      acd132(46)=acd132(34)*acd132(35)
      acd132(47)=acd132(32)*acd132(33)
      acd132(48)=acd132(30)*acd132(31)
      acd132(49)=acd132(28)*acd132(29)
      acd132(50)=acd132(16)*acd132(18)
      acd132(51)=acd132(1)*acd132(4)
      brack=acd132(40)+acd132(41)+acd132(42)+acd132(43)+acd132(44)+acd132(45)+a&
      &cd132(46)+acd132(47)+acd132(48)+acd132(49)+acd132(50)+acd132(51)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd132h4
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(36) :: acd132
      complex(ki) :: brack
      acd132(1)=spvak1e2(iv1)
      acd132(2)=spvae2k2(iv2)
      acd132(3)=abb132(59)
      acd132(4)=spvak1e2(iv2)
      acd132(5)=spvae2k2(iv1)
      acd132(6)=spval5e2(iv2)
      acd132(7)=abb132(20)
      acd132(8)=spvak2e2(iv2)
      acd132(9)=abb132(12)
      acd132(10)=spval4e2(iv2)
      acd132(11)=abb132(61)
      acd132(12)=spvae1e2(iv2)
      acd132(13)=abb132(53)
      acd132(14)=spval5e2(iv1)
      acd132(15)=spvak2e2(iv1)
      acd132(16)=spval4e2(iv1)
      acd132(17)=spvae1e2(iv1)
      acd132(18)=spvae2k1(iv1)
      acd132(19)=abb132(13)
      acd132(20)=spvae2k1(iv2)
      acd132(21)=spvae2l4(iv2)
      acd132(22)=abb132(31)
      acd132(23)=spvae2l5(iv2)
      acd132(24)=abb132(22)
      acd132(25)=spvae2e1(iv2)
      acd132(26)=abb132(14)
      acd132(27)=spvae2l4(iv1)
      acd132(28)=spvae2l5(iv1)
      acd132(29)=spvae2e1(iv1)
      acd132(30)=-acd132(13)*acd132(12)
      acd132(31)=-acd132(11)*acd132(10)
      acd132(32)=acd132(9)*acd132(8)
      acd132(33)=acd132(3)*acd132(4)
      acd132(34)=acd132(6)*acd132(7)
      acd132(30)=acd132(34)+acd132(33)+acd132(32)+acd132(30)+acd132(31)
      acd132(30)=acd132(5)*acd132(30)
      acd132(31)=-acd132(13)*acd132(17)
      acd132(32)=-acd132(11)*acd132(16)
      acd132(33)=acd132(9)*acd132(15)
      acd132(34)=acd132(3)*acd132(1)
      acd132(35)=acd132(14)*acd132(7)
      acd132(31)=acd132(35)+acd132(34)+acd132(33)+acd132(31)+acd132(32)
      acd132(31)=acd132(2)*acd132(31)
      acd132(32)=acd132(26)*acd132(25)
      acd132(33)=acd132(24)*acd132(23)
      acd132(34)=acd132(22)*acd132(21)
      acd132(35)=acd132(19)*acd132(20)
      acd132(32)=acd132(35)+acd132(34)+acd132(32)+acd132(33)
      acd132(32)=acd132(14)*acd132(32)
      acd132(33)=acd132(26)*acd132(29)
      acd132(34)=acd132(24)*acd132(28)
      acd132(35)=acd132(22)*acd132(27)
      acd132(36)=acd132(19)*acd132(18)
      acd132(33)=acd132(36)+acd132(35)+acd132(33)+acd132(34)
      acd132(33)=acd132(6)*acd132(33)
      brack=acd132(30)+acd132(31)+acd132(32)+acd132(33)
   end function brack_3
!---#] function brack_3:
!---#[ function derivative:
   function derivative(mu2,i1,i2) result(numerator)
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd132h4
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
      qshift = -k5
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
end module     p2_gg_httbar_d132h4l1d
