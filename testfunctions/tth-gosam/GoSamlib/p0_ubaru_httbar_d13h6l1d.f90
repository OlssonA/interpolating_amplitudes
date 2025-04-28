module     p0_ubaru_httbar_d13h6l1d
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p0_ubaru_httbar/helicity6d13h6l1d.f90
   ! generator: buildfortran_d.py
   use p0_ubaru_httbar_config, only: ki
   use p0_ubaru_httbar_util, only: cond, d => metric_tensor
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   integer, private :: iv0
   integer, private :: iv1
   integer, private :: iv2
   integer, private :: iv3
   real(ki), dimension(4), private :: qshift
   public :: derivative
contains
!---#[ function brack_1:
   pure function brack_1(Q, mu2) result(brack)
      use p0_ubaru_httbar_model
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_color
      use p0_ubaru_httbar_abbrevd13h6
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(25) :: acd13
      complex(ki) :: brack
      acd13(1)=dotproduct(k2,qshift)
      acd13(2)=dotproduct(qshift,spvak2k1)
      acd13(3)=abb13(15)
      acd13(4)=abb13(12)
      acd13(5)=dotproduct(qshift,qshift)
      acd13(6)=abb13(13)
      acd13(7)=dotproduct(qshift,spval5l4)
      acd13(8)=abb13(11)
      acd13(9)=abb13(9)
      acd13(10)=dotproduct(qshift,spvak2l3)
      acd13(11)=abb13(18)
      acd13(12)=dotproduct(qshift,spvak2l4)
      acd13(13)=abb13(16)
      acd13(14)=dotproduct(qshift,spval3k1)
      acd13(15)=abb13(22)
      acd13(16)=dotproduct(qshift,spval5k1)
      acd13(17)=abb13(17)
      acd13(18)=abb13(10)
      acd13(19)=acd13(3)*acd13(1)
      acd13(20)=acd13(8)*acd13(7)
      acd13(19)=-acd13(9)+acd13(20)+acd13(19)
      acd13(19)=acd13(2)*acd13(19)
      acd13(20)=-acd13(4)*acd13(1)
      acd13(21)=acd13(6)*acd13(5)
      acd13(22)=-acd13(11)*acd13(10)
      acd13(23)=-acd13(13)*acd13(12)
      acd13(24)=-acd13(15)*acd13(14)
      acd13(25)=-acd13(17)*acd13(16)
      brack=acd13(18)+acd13(19)+acd13(20)+acd13(21)+acd13(22)+acd13(23)+acd13(2&
      &4)+acd13(25)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p0_ubaru_httbar_model
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_color
      use p0_ubaru_httbar_abbrevd13h6
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(28) :: acd13
      complex(ki) :: brack
      acd13(1)=k2(iv1)
      acd13(2)=dotproduct(qshift,spvak2k1)
      acd13(3)=abb13(15)
      acd13(4)=abb13(12)
      acd13(5)=qshift(iv1)
      acd13(6)=abb13(13)
      acd13(7)=spvak2k1(iv1)
      acd13(8)=dotproduct(k2,qshift)
      acd13(9)=dotproduct(qshift,spval5l4)
      acd13(10)=abb13(11)
      acd13(11)=abb13(9)
      acd13(12)=spval5l4(iv1)
      acd13(13)=spvak2l3(iv1)
      acd13(14)=abb13(18)
      acd13(15)=spvak2l4(iv1)
      acd13(16)=abb13(16)
      acd13(17)=spval3k1(iv1)
      acd13(18)=abb13(22)
      acd13(19)=spval5k1(iv1)
      acd13(20)=abb13(17)
      acd13(21)=-acd13(3)*acd13(1)
      acd13(22)=-acd13(12)*acd13(10)
      acd13(21)=acd13(22)+acd13(21)
      acd13(21)=acd13(2)*acd13(21)
      acd13(22)=-acd13(8)*acd13(3)
      acd13(23)=-acd13(9)*acd13(10)
      acd13(22)=acd13(11)+acd13(23)+acd13(22)
      acd13(22)=acd13(7)*acd13(22)
      acd13(23)=acd13(4)*acd13(1)
      acd13(24)=acd13(6)*acd13(5)
      acd13(25)=acd13(14)*acd13(13)
      acd13(26)=acd13(16)*acd13(15)
      acd13(27)=acd13(18)*acd13(17)
      acd13(28)=acd13(20)*acd13(19)
      brack=acd13(21)+acd13(22)+acd13(23)-2.0_ki*acd13(24)+acd13(25)+acd13(26)+&
      &acd13(27)+acd13(28)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p0_ubaru_httbar_model
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_color
      use p0_ubaru_httbar_abbrevd13h6
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(13) :: acd13
      complex(ki) :: brack
      acd13(1)=d(iv1,iv2)
      acd13(2)=abb13(13)
      acd13(3)=k2(iv1)
      acd13(4)=spvak2k1(iv2)
      acd13(5)=abb13(15)
      acd13(6)=k2(iv2)
      acd13(7)=spvak2k1(iv1)
      acd13(8)=spval5l4(iv2)
      acd13(9)=abb13(11)
      acd13(10)=spval5l4(iv1)
      acd13(11)=acd13(3)*acd13(4)
      acd13(12)=acd13(6)*acd13(7)
      acd13(11)=acd13(12)+acd13(11)
      acd13(11)=acd13(5)*acd13(11)
      acd13(12)=acd13(8)*acd13(7)
      acd13(13)=acd13(10)*acd13(4)
      acd13(12)=acd13(13)+acd13(12)
      acd13(12)=acd13(9)*acd13(12)
      acd13(13)=acd13(2)*acd13(1)
      brack=acd13(11)+acd13(12)+2.0_ki*acd13(13)
   end function brack_3
!---#] function brack_3:
!---#[ function brack_4:
   pure function brack_4(Q, mu2) result(brack)
      use p0_ubaru_httbar_model
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_color
      use p0_ubaru_httbar_abbrevd13h6
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(1) :: acd13
      complex(ki) :: brack
      brack=0.0_ki
   end function brack_4
!---#] function brack_4:
!---#[ function derivative:
   function derivative(mu2,i1,i2,i3) result(numerator)
      use p0_ubaru_httbar_globalsl1, only: epspow
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_abbrevd13h6
      implicit none
      complex(ki), intent(in) :: mu2
      integer, intent(in), optional :: i1
      integer, intent(in), optional :: i2
      integer, intent(in), optional :: i3
      complex(ki) :: numerator
      complex(ki) :: loc
      integer :: t1
      integer :: deg
      complex(ki), dimension(4), parameter :: Q = (/ (0.0_ki,0.0_ki),(0.0_ki,0.&
      &0_ki),(0.0_ki,0.0_ki),(0.0_ki,0.0_ki)/)
      qshift = k3+k4
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
      if(present(i3)) then
          iv3=i3
          deg=3
      else
          iv3=1
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
      if(deg.eq.3) then
         numerator = cond(epspow.eq.t1,brack_4,Q,mu2)
         return
      end if
   end function derivative
!---#] function derivative:
end module     p0_ubaru_httbar_d13h6l1d
